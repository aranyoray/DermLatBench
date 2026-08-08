"""End-to-end audit orchestration.

Public entry points:
    run_preflight   — discover images, hash, populate the manifest
    run_dedup       — compute pHash, group duplicates + source groups
    run_inference   — run first-pass blinded inference (resumable)
    run_second_pass — re-run blinded inference on low-confidence / invalid rows
    run_full        — all of the above in order
"""

from __future__ import annotations

import json
import platform
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from . import PIPELINE_VERSION, __version__
from .config import AppConfig
from .constants import ReviewFlag
from .db import Database
from .dedup import (
    ImageForDedup,
    compute_orientation_phashes,
    group_exact_duplicates,
    group_near_duplicates,
)
from .hashing import phash_hex
from .inference import Backend, InferenceError, make_backend
from .manifest import discover_images
from .prompts import prompt_hash, prompt_text
from .schema import compute_flags, extract_json, validate_response
from .utils import (
    auto_batch,
    auto_workers,
    detect_hardware,
    ensure_dir,
    format_duration,
    free_disk_gb,
    utc_now_iso,
)

_console = Console()


# --------------------------------------------------------------------- helpers

def _run_id() -> str:
    return uuid.uuid4().hex[:12]


def _record_run_meta(db: Database, cfg: AppConfig, backend: Backend | None) -> None:
    hw = detect_hardware()
    meta = {
        "dermlat_audit_version": __version__,
        "pipeline_version": PIPELINE_VERSION,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hardware": hw,
        "prompt_hash": prompt_hash(),
        "config": cfg.model_dump(mode="json"),
        "backend_name": backend.name if backend else None,
        "backend_model": backend.model_name if backend else None,
        "backend_version": backend.model_version if backend else None,
        "started_at": utc_now_iso(),
    }
    for k, v in meta.items():
        db.set_meta(k, v)
    if not db.get_meta("run_id"):
        db.set_meta("run_id", _run_id())


def _apply_supplied_labels(db: Database, annotations_path: Path | None) -> int:
    """Load supplied laterality labels into the DB, but NEVER expose them to the model.

    Auto-detects a few common CSV column layouts. Missing/unknown columns yield no rows.
    """
    if annotations_path is None or not annotations_path.exists():
        return 0
    import pandas as pd
    if annotations_path.suffix.lower() in (".json",):
        df = pd.read_json(annotations_path)
    else:
        df = pd.read_csv(annotations_path)
    if df.empty:
        return 0
    lower_cols = {c.lower(): c for c in df.columns}
    path_col = next((lower_cols[c] for c in ("path", "filename", "file", "image", "relpath")
                     if c in lower_cols), None)
    label_col = next((lower_cols[c] for c in ("laterality", "label", "side", "class")
                      if c in lower_cols), None)
    if not path_col or not label_col:
        return 0
    n = 0
    for _, row in df.iterrows():
        key = str(row[path_col])
        label = row[label_col]
        if label is None:
            continue
        # supplied paths may be relative — match on relpath OR full path.
        n += _match_and_store_supplied_label(db, key, str(label))
    return n


def _match_and_store_supplied_label(db: Database, key: str, label: str) -> int:
    cur = db._conn.execute(
        "SELECT path FROM images WHERE path=? OR relpath=? LIMIT 1", (key, key)
    ).fetchone()
    if not cur:
        return 0
    db.set_supplied_label(cur["path"], label)
    return 1


# --------------------------------------------------------------------- preflight

def run_preflight(cfg: AppConfig, db: Database) -> dict[str, Any]:
    _console.log(f"[bold]Preflight[/bold] scanning {cfg.paths.images_dir}")
    rows = discover_images(cfg.paths.images_dir, hash_files=True)
    db.upsert_manifest_rows(rows)
    supplied = _apply_supplied_labels(db, cfg.paths.annotations)
    result = {
        "total_discovered": len(rows),
        "readable": sum(1 for r in rows if r.readable),
        "supported": sum(1 for r in rows if r.supported),
        "corrupt": sum(1 for r in rows if r.supported and not r.readable),
        "unsupported": sum(1 for r in rows if not r.supported),
        "supplied_labels_loaded": supplied,
    }
    db.set_meta("preflight", result)
    _console.log(f"  discovered={result['total_discovered']}  readable={result['readable']}  "
                 f"corrupt={result['corrupt']}  unsupported={result['unsupported']}  "
                 f"supplied_labels={supplied}")
    return result


# --------------------------------------------------------------------- dedup

def run_dedup(cfg: AppConfig, db: Database, *, progress_cb: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    _console.log("[bold]Duplicate detection[/bold] computing pHashes + grouping")
    rows = db.all_images()
    items: list[ImageForDedup] = []
    orient: dict[str, list[str]] = {}
    total = len(rows)
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=_console,
        transient=True,
    ) as prog:
        task = prog.add_task("phash", total=total)
        for i, r in enumerate(rows, 1):
            path = r["path"]
            if not r["readable"]:
                items.append(ImageForDedup(path=path, sha256=r["sha256"], phash=None))
            else:
                p = phash_hex(Path(path)) if not r["phash"] else r["phash"]
                if p and p != r["phash"]:
                    db.set_phash(path, p)
                items.append(ImageForDedup(path=path, sha256=r["sha256"], phash=p))
                orients = compute_orientation_phashes(Path(path))
                if orients:
                    orient[path] = orients
            prog.update(task, advance=1)
            if progress_cb:
                progress_cb(i, total)

    dup_groups = group_exact_duplicates(items)
    src_groups = group_near_duplicates(
        items,
        orientation_hashes=orient,
        hamming_threshold=cfg.dedup.near_duplicate_threshold,
        bucket_prefix_bits=cfg.dedup.bucket_prefix_bits,
    )

    now = utc_now_iso()
    with db.transaction() as cur:
        for gid, members in dup_groups.items():
            cur.execute(
                "INSERT OR REPLACE INTO duplicate_groups(duplicate_group_id, sha256, member_count, created_at) VALUES (?,?,?,?)",
                (gid, gid.split("_", 1)[1] if "_" in gid else "", len(members), now),
            )
            for p in members:
                cur.execute("UPDATE images SET duplicate_group_id=? WHERE path=?", (gid, p))
        for gid, members in src_groups.items():
            cur.execute(
                "INSERT OR REPLACE INTO source_groups(source_group_id, member_count, created_at) VALUES (?,?,?)",
                (gid, len(members), now),
            )
            for p in members:
                cur.execute("UPDATE images SET source_group_id=? WHERE path=?", (gid, p))

    summary = {
        "exact_duplicate_groups": len(dup_groups),
        "images_in_exact_duplicate_groups": sum(len(v) for v in dup_groups.values()),
        "source_groups": len(src_groups),
        "singleton_source_groups": sum(1 for v in src_groups.values() if len(v) == 1),
        "multi_member_source_groups": sum(1 for v in src_groups.values() if len(v) >= 2),
        "computed_at": now,
    }
    db.set_meta("dedup", summary)
    _console.log(f"  exact_dup_groups={summary['exact_duplicate_groups']}  "
                 f"source_groups={summary['source_groups']}  "
                 f"multi_member={summary['multi_member_source_groups']}")
    return summary


# --------------------------------------------------------------------- inference

def _persist_prediction(
    db: Database, path: str, backend: Backend, pass_number: int,
    raw_text: str, duration_ms: int, cfg: AppConfig, error: str | None = None,
) -> tuple[bool, list[str], dict[str, Any] | None]:
    parsed: dict[str, Any] | None = None
    flags: list[str] = []
    valid = False
    try:
        parsed = extract_json(raw_text)
        validate_response(parsed)
        valid = True
        flags = compute_flags(parsed, cfg.inference.low_confidence_threshold)
    except (ValueError, jsonschema.ValidationError) as exc:
        flags = [ReviewFlag.INVALID_JSON.value]
        error = error or f"invalid_json_or_schema: {exc}"

    # Only trust typed fields when the response passed schema validation.
    # For invalid responses we store the raw text but null out the typed columns
    # (which downstream code checks for `IS NOT NULL`).
    p = parsed if valid else None
    db.insert_prediction({
        "path": path,
        "model_name": backend.model_name,
        "model_version": backend.model_version,
        "pass_number": pass_number,
        "pipeline_version": PIPELINE_VERSION,
        "prompt_hash": prompt_hash(),
        "raw_response": raw_text,
        "parsed_json": json.dumps(parsed) if parsed is not None else None,
        "laterality": (p or {}).get("laterality"),
        "answerable": int((p or {}).get("answerable", False)) if p else None,
        "confidence": (p or {}).get("confidence"),
        "anatomical_site": (p or {}).get("anatomical_site"),
        "visible_evidence": (p or {}).get("visible_evidence"),
        "ambiguity_reason": (p or {}).get("ambiguity_reason"),
        "review_required": int((p or {}).get("review_required", False)) if p else None,
        "possible_label_error": int((p or {}).get("possible_label_error", False)) if p else None,
        "flags": json.dumps(flags),
        "valid": int(valid),
        "error": error,
        "processed_at": utc_now_iso(),
        "duration_ms": duration_ms,
    })
    return valid, flags, parsed


def _run_one_with_retries(
    backend: Backend, path: Path, prompt: str, retries: int, timeout: int,
) -> tuple[str, int, str | None]:
    last_err: str | None = None
    for attempt in range(1, retries + 1):
        try:
            r = backend.infer(path, prompt, timeout_seconds=timeout)
            return r.raw_text, r.duration_ms, None
        except InferenceError as exc:
            last_err = f"attempt {attempt}: {exc}"
            time.sleep(min(2.0, 0.5 * attempt))
    return "", 0, last_err or "unknown_error"


def _make_progress() -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TextColumn("• {task.fields[status]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=_console,
    )


@dataclass
class _PassCounters:
    processed: int = 0
    failures: int = 0
    flagged: int = 0
    started: float = 0.0


def run_inference(
    cfg: AppConfig, db: Database, backend: Backend, *, pass_number: int = 1,
) -> dict[str, Any]:
    prompt = prompt_text()
    hw = detect_hardware()
    workers = auto_workers(cfg.inference.workers, hw["ram_gb"])
    batch = auto_batch(cfg.inference.batch_size, hw["ram_gb"])

    if pass_number == 1:
        rows = db.readable_images_without_predictions(backend.model_name)
        label = "pass 1"
    else:
        rows = db.images_needing_second_pass(backend.model_name, cfg.inference.low_confidence_threshold)
        label = f"pass {pass_number}"

    total = len(rows)
    _console.log(f"[bold]{label}[/bold] {total} images | model={backend.model_name} "
                 f"| workers={workers} batch={batch}")

    c = _PassCounters(started=time.perf_counter())
    with _make_progress() as prog:
        task = prog.add_task(
            label, total=total, status=f"flagged=0 failures=0 model={backend.model_name[:24]}"
        )
        for r in rows:
            path = Path(r["path"])
            raw, duration_ms, err = _run_one_with_retries(
                backend, path, prompt, cfg.inference.max_json_retries,
                cfg.inference.per_image_timeout_seconds,
            )
            if err and not raw:
                # Persist as a failed prediction row; do NOT silently skip.
                db.insert_prediction({
                    "path": str(path),
                    "model_name": backend.model_name,
                    "model_version": backend.model_version,
                    "pass_number": pass_number,
                    "pipeline_version": PIPELINE_VERSION,
                    "prompt_hash": prompt_hash(),
                    "raw_response": "",
                    "parsed_json": None,
                    "laterality": None,
                    "answerable": None,
                    "confidence": None,
                    "anatomical_site": None,
                    "visible_evidence": None,
                    "ambiguity_reason": None,
                    "review_required": None,
                    "possible_label_error": None,
                    "flags": json.dumps([ReviewFlag.MODEL_ERROR.value]),
                    "valid": 0,
                    "error": err,
                    "processed_at": utc_now_iso(),
                    "duration_ms": duration_ms,
                })
                c.failures += 1
                db.add_to_review_queue(str(path), [ReviewFlag.MODEL_ERROR.value], utc_now_iso())
            else:
                valid, flags, _ = _persist_prediction(
                    db, str(path), backend, pass_number, raw, duration_ms, cfg, err
                )
                if not valid or flags:
                    db.add_to_review_queue(str(path), flags or [ReviewFlag.INVALID_JSON.value], utc_now_iso())
                    c.flagged += 1
            c.processed += 1
            prog.update(
                task, advance=1,
                status=f"flagged={c.flagged} failures={c.failures} model={backend.model_name[:24]}",
            )
    elapsed = time.perf_counter() - c.started
    result = {
        "pass_number": pass_number,
        "processed": c.processed,
        "flagged": c.flagged,
        "failures": c.failures,
        "seconds": round(elapsed, 2),
        "throughput_img_per_s": round(c.processed / elapsed, 3) if elapsed > 0 else 0,
    }
    key = f"inference_pass_{pass_number}"
    db.set_meta(key, result)
    _console.log(f"  {label}: processed={c.processed} flagged={c.flagged} "
                 f"failures={c.failures} in {format_duration(elapsed)}")
    return result


# --------------------------------------------------------------------- second pass / disagreement flags

def flag_disagreements(db: Database) -> int:
    """Add DISAGREEMENT_BETWEEN_PASSES flag where pass 1 and pass 2 disagree on label."""
    rows = db._conn.execute(
        """
        SELECT p1.path, p1.laterality AS l1, p2.laterality AS l2
        FROM predictions p1
        JOIN predictions p2 ON p1.path = p2.path AND p1.model_name = p2.model_name
        WHERE p1.pass_number = 1 AND p2.pass_number = 2
        """
    ).fetchall()
    n = 0
    for r in rows:
        if r["l1"] != r["l2"]:
            existing = db._conn.execute(
                "SELECT reasons FROM review_queue WHERE path=?", (r["path"],)
            ).fetchone()
            reasons: list[str] = json.loads(existing["reasons"]) if existing else []
            reasons.append(ReviewFlag.DISAGREEMENT_BETWEEN_PASSES.value)
            db.add_to_review_queue(r["path"], reasons, utc_now_iso())
            n += 1
    return n


# --------------------------------------------------------------------- top-level

def run_full(cfg: AppConfig, output_dir: Path) -> dict[str, Any]:
    ensure_dir(output_dir)
    db_path = output_dir / "audit.sqlite"
    with Database(db_path) as db:
        backend = make_backend(cfg)
        _record_run_meta(db, cfg, backend)
        # Preflight (idempotent — safe on resume).
        _ = run_preflight(cfg, db)
        # Dedup (idempotent).
        _ = run_dedup(cfg, db)
        # Inference pass 1 (resumable — only images without predictions).
        pass1 = run_inference(cfg, db, backend, pass_number=1)
        # Second pass on low-conf + invalid rows.
        pass2 = run_inference(cfg, db, backend, pass_number=2)
        n_disagreements = flag_disagreements(db)
        db.set_meta("finished_at", utc_now_iso())
        return {"pass1": pass1, "pass2": pass2, "disagreements_flagged": n_disagreements}


def preflight_hardware_report(cfg: AppConfig, output_dir: Path) -> dict[str, Any]:
    """Called by the CLI before starting; prints a hardware summary."""
    hw = detect_hardware()
    free = free_disk_gb(output_dir if output_dir.exists() else Path.cwd())
    _console.log(f"[bold]Hardware[/bold] {hw['apple_chip'] or hw['platform']}  "
                 f"RAM={hw['ram_gb']} GB  cores={hw['cpu_count']}  free_disk={free} GB")
    if hw["ram_gb"] < 8:
        _console.log("[yellow]warning[/yellow] very low RAM; consider the mock or ollama backend.")
    return hw | {"free_disk_gb": free}


__all__ = [
    "Database",
    "flag_disagreements",
    "preflight_hardware_report",
    "run_dedup",
    "run_full",
    "run_inference",
    "run_preflight",
]
