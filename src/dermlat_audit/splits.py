"""Leakage-safe train/val/test splits by source_group_id.

- Two derivatives of the same source image (mirrors, rotations, crops, resizes) share a
  source_group_id and therefore always land in the SAME split.
- validate_no_leakage() re-checks this invariant and raises if it is ever violated.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from .config import SplitsConfig


@dataclass
class SplitAssignment:
    path: str
    source_group_id: str
    split: str  # "train" | "val" | "test"


class LeakageError(RuntimeError):
    """Raised when two members of the same source_group_id are in different splits."""


def make_splits(rows: Iterable[dict], cfg: SplitsConfig) -> list[SplitAssignment]:
    """Assign every row to a split by hashing on source_group_id.

    Rows lacking a source_group_id are treated as singleton groups keyed on their path.
    """
    rows = list(rows)
    # gather unique groups in a stable order
    groups: dict[str, list[str]] = {}
    for r in rows:
        gid = r.get("source_group_id") or f"__solo__:{r['path']}"
        groups.setdefault(gid, []).append(r["path"])

    unique_groups = sorted(groups)
    rng = random.Random(cfg.random_seed)
    rng.shuffle(unique_groups)
    n = len(unique_groups)
    n_train = round(n * cfg.train)
    n_val = round(n * cfg.val)
    # Assign the remainder to test to guarantee coverage.
    train_g = set(unique_groups[:n_train])
    val_g = set(unique_groups[n_train:n_train + n_val])
    # test_g gets whatever remains

    assignments: list[SplitAssignment] = []
    for gid, paths in groups.items():
        if gid in train_g:
            split = "train"
        elif gid in val_g:
            split = "val"
        else:
            split = "test"
        display_gid = gid.split(":", 1)[1] if gid.startswith("__solo__:") else gid
        for p in paths:
            assignments.append(SplitAssignment(path=p, source_group_id=display_gid, split=split))
    return assignments


def validate_no_leakage(assignments: list[SplitAssignment]) -> None:
    """Raise LeakageError if any source_group_id spans multiple splits."""
    by_group: dict[str, set[str]] = {}
    for a in assignments:
        by_group.setdefault(a.source_group_id, set()).add(a.split)
    leaks = {gid: splits for gid, splits in by_group.items() if len(splits) > 1}
    if leaks:
        preview = list(leaks.items())[:5]
        raise LeakageError(
            f"{len(leaks)} source_group_id(s) span multiple splits. Examples: {preview}"
        )


def assignments_to_dataframe(assignments: list[SplitAssignment]) -> pd.DataFrame:
    return pd.DataFrame([a.__dict__ for a in assignments]).sort_values("path").reset_index(drop=True)
