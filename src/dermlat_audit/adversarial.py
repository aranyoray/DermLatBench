"""Adversarial-blinding experiment.

Runs the pipeline twice on the same images:
  (a) blinded: the standard BLINDED_LATERALITY_PROMPT (pixels only).
  (b) leaky: the same prompt with the FILENAME deliberately appended.

The comparison quantifies how much unblinded metadata inflates apparent accuracy —
the key methodological claim of the paper.

Only the mock/ollama backend is safe here (they can be instrumented to accept a
per-image prompt override). MLX-VLM is also supported via the standard interface
because the leaky prompt is just a longer string.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import jsonschema

from . import PIPELINE_VERSION
from .config import AppConfig
from .constants import UNILATERAL_LABELS, ReviewFlag
from .db import Database
from .inference import Backend, InferenceError
from .prompts import BLINDED_LATERALITY_PROMPT
from .schema import compute_flags, extract_json, validate_response
from .utils import ensure_dir, utc_now_iso

LEAKY_SUFFIX = (
    "\n\nADDITIONAL CONTEXT (INTENTIONALLY LEAKED FOR AUDIT — DO NOT REPLY WITH IT): "
    "the filename of this image is: {filename}\n"
    "the parent folder name is: {folder}"
)


def leaky_prompt(filename: str, folder: str) -> str:
    return BLINDED_LATERALITY_PROMPT + LEAKY_SUFFIX.format(filename=filename, folder=folder)


def leaky_prompt_hash() -> str:
    return hashlib.sha256(
        (BLINDED_LATERALITY_PROMPT + LEAKY_SUFFIX).encode("utf-8")
    ).hexdigest()


def _persist(
    db: Database, path: str, backend: Backend, raw_text: str, duration_ms: int,
    cfg: AppConfig, pass_number: int, prompt_hash_value: str, error: str | None,
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
    p = parsed if valid else None
    db.insert_prediction({
        "path": path,
        "model_name": backend.model_name + "+leaky",
        "model_version": backend.model_version,
        "pass_number": pass_number,
        "pipeline_version": PIPELINE_VERSION,
        "prompt_hash": prompt_hash_value,
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


def run_leaky_pass(cfg: AppConfig, db: Database, backend: Backend) -> dict[str, Any]:
    """Run inference with filename+folder leaked into the prompt (single pass)."""
    prompt_h = leaky_prompt_hash()
    rows = [r for r in db.all_images() if r["readable"]]
    n_ok = 0
    n_err = 0
    started = time.perf_counter()
    for r in rows:
        path = Path(r["path"])
        prompt = leaky_prompt(filename=path.name, folder=path.parent.name)
        try:
            res = backend.infer(path, prompt, timeout_seconds=cfg.inference.per_image_timeout_seconds)
            _persist(db, str(path), backend, res.raw_text, res.duration_ms,
                     cfg, pass_number=1, prompt_hash_value=prompt_h, error=None)
            n_ok += 1
        except InferenceError as exc:
            _persist(db, str(path), backend, "", 0, cfg,
                     pass_number=1, prompt_hash_value=prompt_h, error=str(exc))
            n_err += 1
    elapsed = time.perf_counter() - started
    result = {
        "processed": n_ok + n_err,
        "failures": n_err,
        "seconds": round(elapsed, 3),
        "prompt_hash": prompt_h,
        "leaky_model_name": backend.model_name + "+leaky",
    }
    db.set_meta("adversarial_leaky_pass", result)
    return result


def compare_blinded_vs_leaky(db: Database) -> dict[str, Any]:
    """Score both models on rows with supplied unilateral labels."""
    supplied = {r["path"]: r["supplied_label"] for r in db._conn.execute(
        "SELECT path, supplied_label FROM images WHERE supplied_label IS NOT NULL"
    )}
    unilat = {p: s.strip().upper() for p, s in supplied.items()
              if s and s.strip().upper() in UNILATERAL_LABELS}

    def _acc(model_name: str) -> dict[str, Any]:
        preds = list(db._conn.execute(
            "SELECT path, laterality FROM predictions WHERE pass_number=1 AND model_name=?",
            (model_name,),
        ))
        n = 0
        correct = 0
        for r in preds:
            if r["path"] not in unilat or not r["laterality"]:
                continue
            n += 1
            if r["laterality"] == unilat[r["path"]]:
                correct += 1
        return {"n": n, "correct": correct,
                "accuracy": (correct / n) if n else 0.0}

    model_names = [r["model_name"] for r in db._conn.execute(
        "SELECT DISTINCT model_name FROM predictions"
    )]
    blinded_names = [m for m in model_names if not m.endswith("+leaky")]
    leaky_names = [m for m in model_names if m.endswith("+leaky")]
    if not blinded_names or not leaky_names:
        return {"note": "need both a blinded and a leaky pass",
                "blinded_names": blinded_names, "leaky_names": leaky_names}
    blinded = _acc(blinded_names[0])
    leaky = _acc(leaky_names[0])
    inflation = leaky["accuracy"] - blinded["accuracy"]
    result = {
        "blinded_model": blinded_names[0],
        "leaky_model": leaky_names[0],
        "blinded": blinded,
        "leaky": leaky,
        "accuracy_inflation_from_leak": inflation,
        "relative_inflation_pct": (inflation / blinded["accuracy"] * 100)
                                   if blinded["accuracy"] else None,
    }
    db.set_meta("adversarial_comparison", result)
    return result


def write_adversarial_report(comparison: dict[str, Any], out_dir: Path) -> Path:
    ensure_dir(out_dir)
    p = out_dir / "adversarial_blinding.json"
    p.write_text(json.dumps(comparison, indent=2))
    return p
