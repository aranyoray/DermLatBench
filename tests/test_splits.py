from __future__ import annotations

import pytest

from dermlat_audit.config import SplitsConfig
from dermlat_audit.splits import LeakageError, make_splits, validate_no_leakage


def _rows(n, groups_per=1):
    """Generate rows with source_group_ids so groups have known members."""
    out = []
    for gi in range(n):
        for j in range(groups_per):
            out.append({"path": f"/img/{gi}_{j}.png", "source_group_id": f"g_{gi:04d}"})
    return out


def test_split_ratios_close(tmp_path):
    rows = _rows(1000, groups_per=1)
    cfg = SplitsConfig()
    a = make_splits(rows, cfg)
    counts = {"train": 0, "val": 0, "test": 0}
    for x in a:
        counts[x.split] += 1
    total = sum(counts.values())
    assert abs(counts["train"] / total - 0.70) < 0.05
    assert abs(counts["val"] / total - 0.15) < 0.05
    assert abs(counts["test"] / total - 0.15) < 0.05


def test_no_leakage_within_source_group():
    rows = _rows(100, groups_per=4)  # 4 derivatives per source
    a = make_splits(rows, SplitsConfig())
    validate_no_leakage(a)  # must not raise


def test_leakage_detector_raises_on_bad_input():
    a = make_splits(_rows(50, groups_per=3), SplitsConfig())
    # Corrupt one member of a group into a different split.
    a[0].split = "train" if a[0].split != "train" else "val"
    with pytest.raises(LeakageError):
        validate_no_leakage(a)


def test_solo_rows_get_singleton_groups():
    rows = [{"path": f"/p/{i}.png", "source_group_id": None} for i in range(20)]
    a = make_splits(rows, SplitsConfig())
    # Every row assigned; no leakage even though group_ids were None.
    validate_no_leakage(a)
    assert len(a) == 20
