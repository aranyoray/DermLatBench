"""dermlat-audit CLI (Typer).

Sub-commands:
    doctor                Detect hardware, deps, and print planned batch sizes.
    run                   Run the full audit pipeline.
    resume                Same as run — everything is idempotent.
    report                Rebuild CSVs, metrics.json, and audit_report.md.
    review                Launch the two-reviewer Streamlit app.
    verify                Print the claim-verification table.
    splits                Emit leakage-safe train/val/test splits.
    safe-add              Show what git would stage. Aborts if data files.
    validate-injections   Inject known errors + score the auditor.
    adversarial-blinding  Quantify accuracy inflation from leaked metadata.
    build-paper           Regenerate LaTeX numbers.tex + tables from metrics.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .adversarial import (
    compare_blinded_vs_leaky,
    run_leaky_pass,
    write_adversarial_report,
)
from .config import AppConfig, load_config
from .db import Database
from .inference import make_backend
from .injection import (
    build_corrupted_dataset,
    score_injection_detection,
    write_injection_report,
)
from .latex import build_paper
from .pipeline import preflight_hardware_report, run_dedup, run_full, run_preflight
from .report import build_all

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="dermlat-audit — local audit pipeline for DermLatBench.")
_console = Console()


def _load(
    config_path: Path | None,
    images_dir: Path | None,
    annotations: Path | None,
    output_dir: Path | None,
    backend: str | None,
) -> AppConfig:
    overrides = {
        "paths.images_dir": str(images_dir) if images_dir else None,
        "paths.annotations": str(annotations) if annotations else None,
        "paths.output_dir": str(output_dir) if output_dir else None,
        "backend": backend,
    }
    cfg = load_config(config_path, overrides={k: v for k, v in overrides.items() if v is not None})
    if not cfg.paths.images_dir or not cfg.paths.output_dir:
        typer.secho("--images and --output are required (or set in config.yaml / env).",
                    fg=typer.colors.RED)
        raise typer.Exit(code=2)
    return cfg


_CFG = typer.Option(None, "--config", "-c", help="Path to config.yaml.")
_IMG = typer.Option(None, "--images", help="Root directory of images (recursive).")
_ANN = typer.Option(None, "--annotations", help="Optional CSV/JSON of supplied annotations.")
_OUT = typer.Option(None, "--output", "-o", help="Output directory.")
_BAK = typer.Option(None, "--backend", "-b", help="mlx | ollama | mock")


@app.command()
def doctor(
    config: Path | None = _CFG,
    images: Path | None = _IMG,
    annotations: Path | None = _ANN,
    output: Path | None = _OUT,
    backend: str | None = _BAK,
) -> None:
    """Hardware / config sanity check. Prints nothing sensitive."""
    cfg = _load(config, images, annotations, output, backend)
    hw = preflight_hardware_report(cfg, cfg.paths.output_dir)
    tbl = Table(title="dermlat-audit doctor")
    for k, v in hw.items():
        tbl.add_row(str(k), str(v))
    tbl.add_row("backend", cfg.backend)
    tbl.add_row("images_dir", str(cfg.paths.images_dir))
    tbl.add_row("output_dir", str(cfg.paths.output_dir))
    tbl.add_row("annotations", str(cfg.paths.annotations))
    _console.print(tbl)


@app.command()
def run(
    config: Path | None = _CFG,
    images: Path | None = _IMG,
    annotations: Path | None = _ANN,
    output: Path | None = _OUT,
    backend: str | None = _BAK,
    report_after: bool = typer.Option(True, "--report/--no-report",
                                      help="Build metrics.json + audit_report.md after inference."),
) -> None:
    """Run the full audit pipeline (idempotent — safe to re-run)."""
    cfg = _load(config, images, annotations, output, backend)
    result = run_full(cfg, cfg.paths.output_dir)
    _console.print_json(json.dumps(result, default=str))
    if report_after:
        _console.log("Building report artefacts…")
        summary = build_all(cfg, cfg.paths.output_dir)
        _console.log(f"report: {summary['report_md']}")
    _console.log(f"[green]done[/green] — output at {cfg.paths.output_dir}")
    _console.log(f"resume with: dermlat-audit resume --output {cfg.paths.output_dir}")


@app.command()
def resume(
    config: Path | None = _CFG,
    images: Path | None = _IMG,
    annotations: Path | None = _ANN,
    output: Path | None = _OUT,
    backend: str | None = _BAK,
) -> None:
    """Resume an interrupted run. Same as `run`."""
    run(config=config, images=images, annotations=annotations,
        output=output, backend=backend, report_after=True)


@app.command()
def preflight(
    config: Path | None = _CFG,
    images: Path | None = _IMG,
    annotations: Path | None = _ANN,
    output: Path | None = _OUT,
    backend: str | None = _BAK,
) -> None:
    """Only run preflight + inventory + dedup. Skips inference."""
    cfg = _load(config, images, annotations, output, backend)
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    with Database(cfg.paths.output_dir / "audit.sqlite") as db:
        _ = run_preflight(cfg, db)
        _ = run_dedup(cfg, db)


@app.command()
def report(
    config: Path | None = _CFG,
    images: Path | None = _IMG,
    annotations: Path | None = _ANN,
    output: Path | None = _OUT,
    backend: str | None = _BAK,
) -> None:
    """(Re)generate metrics.json, all CSVs, and audit_report.md from the existing DB."""
    cfg = _load(config, images, annotations, output, backend)
    summary = build_all(cfg, cfg.paths.output_dir)
    _console.log(f"report: {summary['report_md']}")
    _console.log(f"metrics: {cfg.paths.output_dir / 'metrics.json'}")


@app.command()
def review(
    config: Path | None = _CFG,
    output: Path | None = _OUT,
    port: int | None = typer.Option(None, "--port", "-p"),
) -> None:
    """Launch the local Streamlit reviewer UI."""
    cfg = _load(config, None, None, output, None)
    p = port or cfg.review.port
    env = os.environ.copy()
    env["DERMLAT_OUTPUT_DIR"] = str(cfg.paths.output_dir)
    env["DERMLAT_BLIND_R2"] = "1" if cfg.review.blind_reviewer_2 else "0"
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(Path(__file__).parent / "reviewer_app.py"),
        "--server.headless", "true",
        "--server.port", str(p),
    ]
    _console.log(f"launching reviewer at http://127.0.0.1:{p}")
    subprocess.run(cmd, env=env, check=False)


@app.command()
def verify(
    config: Path | None = _CFG,
    images: Path | None = _IMG,
    annotations: Path | None = _ANN,
    output: Path | None = _OUT,
    backend: str | None = _BAK,
) -> None:
    """Print the claim-verification table."""
    cfg = _load(config, images, annotations, output, backend)
    metrics_file = cfg.paths.output_dir / "metrics.json"
    if not metrics_file.exists():
        summary = build_all(cfg, cfg.paths.output_dir)
        metrics = summary["metrics"]
    else:
        metrics = json.loads(metrics_file.read_text())
    tbl = Table(title="Claim verification")
    tbl.add_column("claim")
    tbl.add_column("claimed")
    tbl.add_column("observed")
    tbl.add_column("status")
    tbl.add_column("notes")
    for c in metrics.get("claim_verification", []):
        tbl.add_row(c["claim"], str(c["claim_value"]), str(c["observed_value"]),
                    c["status"], c["notes"])
    _console.print(tbl)


@app.command()
def splits(
    config: Path | None = _CFG,
    images: Path | None = _IMG,
    annotations: Path | None = _ANN,
    output: Path | None = _OUT,
    backend: str | None = _BAK,
) -> None:
    """Emit leakage_safe_splits.csv and print counts."""
    cfg = _load(config, images, annotations, output, backend)
    summary = build_all(cfg, cfg.paths.output_dir)
    m = summary["metrics"]["leakage_check"]
    _console.log(f"leakage_check ok={m['ok']} rows={m['n_rows']} note={m['note']}")


@app.command("safe-add")
def safe_add(
    output: Path | None = _OUT,
) -> None:
    """List what git would stage. Warn if the output dir shows up."""
    try:
        r = subprocess.run(["git", "status", "--porcelain=v1"], capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        _console.log("[yellow]not a git repo — nothing to check[/yellow]")
        return
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    trouble = []
    output_str = str(output) if output else None
    # Whitelist known-safe repo files that would otherwise match substrings.
    safe_names = {".env.example", "config.example.yaml"}
    for ln in lines:
        p = ln[3:]
        pl = p.lower()
        base = p.rsplit("/", 1)[-1]
        if base in safe_names or pl.startswith("samples/"):
            risky = False
        else:
            risky = (
                "audit.sqlite" in pl
                or pl.endswith(".env")
                or pl.endswith(".env.local")
                or "/images/" in pl
                or pl == "images"
                or pl.startswith("images/")
                or "/annotations" in pl
                or pl.startswith("annotations")
                or "manuscript" in pl
            )
        if output_str and p.startswith(os.path.relpath(output_str)):
            risky = True
        trouble.append((p, risky))
    tbl = Table(title="git status")
    tbl.add_column("file")
    tbl.add_column("risky")
    for p, risky in trouble:
        tbl.add_row(p, "[red]YES[/red]" if risky else "[green]no[/green]")
    _console.print(tbl)
    if any(r for _, r in trouble):
        _console.log("[red]refuse to auto-add[/red] — inspect risky files above.")
        raise typer.Exit(code=1)
    _console.log("all files look safe to add.")


# --------------------------------------------------------------------- paper experiments

@app.command("validate-injections")
def validate_injections(
    config: Path | None = _CFG,
    images: Path | None = _IMG,
    annotations: Path | None = _ANN,
    output: Path | None = _OUT,
    backend: str | None = _BAK,
    n_flips: int = typer.Option(50, "--n-flips", help="Number of label flips to inject."),
    n_duplicates: int = typer.Option(50, "--n-duplicates",
                                     help="Number of near-duplicate images to inject."),
    seed: int = typer.Option(20260808, "--seed"),
) -> None:
    """Inject known label flips + duplicates into a copy of the dataset, then score
    the auditor's recall/precision. Writes injection_report.{json,csv} into --output."""
    cfg = _load(config, images, annotations, output, backend)
    if cfg.paths.annotations is None:
        typer.secho("--annotations is required for injection validation.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=2)

    corrupt_root = cfg.paths.output_dir / "injection_workspace"
    corrupt_images, corrupt_ann, gt = build_corrupted_dataset(
        cfg.paths.images_dir, cfg.paths.annotations, corrupt_root,
        n_flips=n_flips, n_duplicates=n_duplicates, seed=seed,
    )
    _console.log(f"corrupted dataset at {corrupt_root} "
                 f"(flips={len(gt.label_flips)}, dup_pairs={len(gt.duplicate_pairs)})")

    # Run the full pipeline over the corrupted dataset in a sibling output dir.
    corrupt_out = cfg.paths.output_dir / "injection_run"
    from copy import deepcopy
    cfg2 = deepcopy(cfg)
    cfg2.paths.images_dir = corrupt_images
    cfg2.paths.annotations = corrupt_ann
    cfg2.paths.output_dir = corrupt_out
    _ = run_full(cfg2, corrupt_out)

    with Database(corrupt_out / "audit.sqlite") as db:
        scores = score_injection_detection(db, gt)
        db.set_meta("injection_scores", scores)
        db.set_meta("injection_ground_truth_summary", {
            "n_flips": len(gt.label_flips),
            "n_duplicate_pairs": len(gt.duplicate_pairs),
        })
    p = write_injection_report(scores, corrupt_out)
    _console.log(f"injection report: {p}")
    _console.print_json(json.dumps(scores, default=str))


@app.command("adversarial-blinding")
def adversarial_blinding(
    config: Path | None = _CFG,
    images: Path | None = _IMG,
    annotations: Path | None = _ANN,
    output: Path | None = _OUT,
    backend: str | None = _BAK,
) -> None:
    """Run a leaky-prompt pass on the same images already audited and compare
    against the blinded pass. Requires that `dermlat-audit run` finished first."""
    cfg = _load(config, images, annotations, output, backend)
    db_path = cfg.paths.output_dir / "audit.sqlite"
    if not db_path.exists():
        typer.secho(f"No audit.sqlite at {db_path}. Run `dermlat-audit run` first.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=2)
    with Database(db_path) as db:
        b = make_backend(cfg)
        leaky = run_leaky_pass(cfg, db, b)
        _console.log(f"leaky pass done: {leaky['processed']} images, {leaky['failures']} failures")
        comparison = compare_blinded_vs_leaky(db)
        p = write_adversarial_report(comparison, cfg.paths.output_dir)
    _console.log(f"adversarial report: {p}")
    _console.print_json(json.dumps(comparison, default=str))


@app.command("build-paper")
def build_paper_cmd(
    config: Path | None = _CFG,
    images: Path | None = _IMG,
    annotations: Path | None = _ANN,
    output: Path | None = _OUT,
    backend: str | None = _BAK,
) -> None:
    """Regenerate LaTeX numbers.tex and per-run tables from metrics.json."""
    cfg = _load(config, images, annotations, output, backend)
    metrics_path = cfg.paths.output_dir / "metrics.json"
    if not metrics_path.exists():
        # Ensure a fresh metrics.json exists.
        build_all(cfg, cfg.paths.output_dir)
    metrics = json.loads((cfg.paths.output_dir / "metrics.json").read_text())
    result = build_paper(metrics, cfg.paths.output_dir)
    _console.log(f"paper dir : {result['paper_dir']}")
    _console.log(f"numbers   : {result['numbers_tex']}")
    for t in result["tables"]:
        _console.log(f"table     : {t}")
    for c in result["template_files"]:
        _console.log(f"template  : {c}")


if __name__ == "__main__":
    app()
