from __future__ import annotations

import json
from pathlib import Path

from dermlat_audit.config import AppConfig, PathsConfig
from dermlat_audit.pipeline import run_full
from dermlat_audit.report import build_all


def test_end_to_end_smoke(tmp_path: Path, synthetic_dataset):
    out = tmp_path / "out"
    cfg = AppConfig(
        paths=PathsConfig(
            images_dir=synthetic_dataset["images_dir"],
            annotations=synthetic_dataset["annotations_csv"],
            output_dir=out,
        ),
        backend="mock",
    )
    r = run_full(cfg, out)
    assert r["pass1"]["processed"] > 0
    build_all(cfg, out)
    # Every required output file exists.
    for name in (
        "complete_manifest.csv", "automated_predictions.csv",
        "flagged_human_review.csv", "reviewer_1.csv", "reviewer_2.csv",
        "adjudicated_labels.csv", "duplicate_groups.csv", "source_groups.csv",
        "leakage_safe_splits.csv", "metrics.json", "audit_report.md",
        "run_manifest.json",
    ):
        assert (out / name).exists(), f"missing {name}"

    # metrics.json is well-formed
    metrics = json.loads((out / "metrics.json").read_text())
    assert "denominators" in metrics
    assert "label_distribution" in metrics
    assert "claim_verification" in metrics

    # audit_report.md contains the required final statement
    report = (out / "audit_report.md").read_text()
    assert "Processed" in report
    assert "human-review queue" in report
    assert "Dataset-wide claims are" in report

    # leakage-safe splits should pass validation
    assert metrics["leakage_check"]["ok"] is True

    # tables + at least one figure directory exists
    assert (out / "tables" / "table_1_denominators.csv").exists()
