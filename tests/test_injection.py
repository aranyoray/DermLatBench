from __future__ import annotations

import csv
from pathlib import Path

from dermlat_audit.config import AppConfig, PathsConfig
from dermlat_audit.db import Database
from dermlat_audit.injection import (
    build_corrupted_dataset,
    inject_duplicates,
    inject_label_flips,
    score_injection_detection,
    write_injection_report,
)
from dermlat_audit.pipeline import run_full


def _annotations_csv(tmp_path: Path) -> Path:
    p = tmp_path / "ann.csv"
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "laterality"])
        w.writeheader()
        for i in range(20):
            side = "LEFT" if i % 2 == 0 else "RIGHT"
            w.writerow({"path": f"img_{i}.png", "laterality": side})
    return p


def test_inject_label_flips_creates_expected_flips(tmp_path: Path):
    src = _annotations_csv(tmp_path)
    dst = tmp_path / "out.csv"
    flips = inject_label_flips(src, dst, n_flips=5, seed=1)
    assert len(flips) == 5
    # Each corrupted label is different from the original.
    for _path, (orig, corrupted) in flips.items():
        assert orig != corrupted
        assert orig in ("LEFT", "RIGHT")
        assert corrupted in ("LEFT", "RIGHT")
    # And the written CSV has the corrupted labels.
    written = {r["path"]: r["laterality"] for r in csv.DictReader(dst.open())}
    for path, (_orig, corrupted) in flips.items():
        assert written[path] == corrupted


def test_inject_duplicates_creates_files(tmp_path: Path, synthetic_dataset):
    pairs = inject_duplicates(synthetic_dataset["images_dir"],
                              n_duplicates=6, seed=42)
    assert len(pairs) == 6
    for _src, dst, tf in pairs:
        assert Path(dst).exists()
        assert tf in ("mirror", "rot90", "rot180", "rot270", "reencode")


def test_end_to_end_injection_scoring(tmp_path: Path, synthetic_dataset):
    """Full flow: corrupt copy → run pipeline → score detection."""
    root = tmp_path / "workspace"
    corrupt_images, corrupt_ann, gt = build_corrupted_dataset(
        synthetic_dataset["images_dir"], synthetic_dataset["annotations_csv"], root,
        n_flips=4, n_duplicates=4, seed=7,
    )
    assert len(gt.label_flips) == 4
    assert len(gt.duplicate_pairs) == 4

    out = tmp_path / "run"
    cfg = AppConfig(paths=PathsConfig(images_dir=corrupt_images,
                                       annotations=corrupt_ann,
                                       output_dir=out),
                    backend="mock")
    run_full(cfg, out)
    with Database(out / "audit.sqlite") as db:
        scores = score_injection_detection(db, gt)
    # Duplicate detection: at least one of the injected pairs should match.
    assert scores["duplicate_detection"]["known_duplicates"] == 4
    assert scores["duplicate_detection"]["tp"] >= 1
    # Label-flip scoring returns valid structure with correct known_flips.
    assert scores["label_flip_detection"]["known_flips"] == 4
    # Report is writable.
    p = write_injection_report(scores, out)
    assert p.exists()
    lines = list(csv.DictReader(p.open()))
    assert {r["check"] for r in lines} == {"label_flip_detection", "duplicate_detection"}
