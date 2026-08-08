from __future__ import annotations

import json
from pathlib import Path

from dermlat_audit.config import AppConfig, PathsConfig
from dermlat_audit.latex import build_paper, write_numbers_tex
from dermlat_audit.paper_figures import (
    calibration_metrics,
    coverage_accuracy_curve,
)
from dermlat_audit.pipeline import run_full
from dermlat_audit.report import build_all


def test_calibration_metrics_perfect():
    # 100% confidence, all correct → ECE 0.
    cm = calibration_metrics([100] * 10, [True] * 10, n_bins=5)
    assert abs(cm["ece"]) < 1e-9
    assert cm["mce"] < 1e-9


def test_calibration_metrics_overconfident():
    # 90% confidence, 10% correct → ECE ~0.8.
    cm = calibration_metrics([90] * 10, [False] * 9 + [True], n_bins=1)
    assert cm["ece"] > 0.5


def test_coverage_accuracy_curve_sorted_desc():
    # Confidence descending → accuracy at coverage 1/n should reflect top item.
    curve = coverage_accuracy_curve([90, 50, 10], [True, False, True])
    assert curve[0]["coverage"] == 1 / 3
    assert curve[0]["accuracy"] == 1.0  # top-confidence item is correct
    assert curve[-1]["coverage"] == 1.0


def test_build_paper_writes_numbers_and_tables(tmp_path: Path, synthetic_dataset):
    cfg = AppConfig(paths=PathsConfig(images_dir=synthetic_dataset["images_dir"],
                                       annotations=synthetic_dataset["annotations_csv"],
                                       output_dir=tmp_path),
                    backend="mock")
    run_full(cfg, tmp_path)
    build_all(cfg, tmp_path)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    r = build_paper(metrics, tmp_path)
    paper_dir = Path(r["paper_dir"])
    assert (paper_dir / "numbers.tex").exists()
    assert (paper_dir / "tables" / "denominators.tex").exists()
    assert (paper_dir / "tables" / "claims.tex").exists()
    text = (paper_dir / "numbers.tex").read_text()
    for macro in ("\\NTotal", "\\NProcessed", "\\SuppliedAgreement",
                  "\\AccUnilateral", "\\ECE", "\\PublicationReady"):
        assert macro in text, f"missing macro {macro}"


def test_figures_are_written(tmp_path: Path, synthetic_dataset):
    cfg = AppConfig(paths=PathsConfig(images_dir=synthetic_dataset["images_dir"],
                                       annotations=synthetic_dataset["annotations_csv"],
                                       output_dir=tmp_path),
                    backend="mock")
    run_full(cfg, tmp_path)
    build_all(cfg, tmp_path)
    figures_dir = tmp_path / "figures"
    # At least the confusion matrix and coverage-accuracy figs should exist
    # (source-group histogram + reliability diagram require enough data).
    fig_names = {f.name for f in figures_dir.glob("*.png")}
    assert any("fig_" in n for n in fig_names)


def test_numbers_tex_survives_missing_metrics(tmp_path: Path):
    minimal_metrics = {"denominators": {}, "label_distribution": {}, "answerable_split": {}}
    p = write_numbers_tex(minimal_metrics, tmp_path / "paper")
    assert p.exists()
    text = p.read_text()
    # Should include n/a for missing metrics rather than crashing.
    assert "\\newcommand{\\NTotal}" in text
