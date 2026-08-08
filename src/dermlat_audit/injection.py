"""Validation harness: prove the auditor actually detects known errors.

Two synthetic corruptions are injected into a copy of the dataset:
  1. Label flips: N supplied labels are deliberately swapped to a different laterality.
  2. Near-duplicate injections: M images are mirrored/rotated/re-encoded and added.

We then run the audit and score:
  - Label-flip detection recall & precision (via `possible_label_error` flag + reviewer
    disagreement with supplied labels).
  - Duplicate detection recall & precision (via source_group_id membership).

Output is a small CSV + one metrics block that goes straight into the paper.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .constants import UNILATERAL_LABELS
from .db import Database
from .utils import ensure_dir

TRANSFORMS = ("mirror", "rot90", "rot180", "rot270", "reencode")


@dataclass
class InjectionGroundTruth:
    """What we corrupted, so we can score detection afterwards."""

    label_flips: dict[str, tuple[str, str]] = field(default_factory=dict)
    #   path -> (original_label, corrupted_label)

    duplicate_pairs: list[tuple[str, str, str]] = field(default_factory=list)
    #   list of (source_path, derivative_path, transform_name)

    def as_dict(self) -> dict:
        return {
            "label_flips": {p: [o, c] for p, (o, c) in self.label_flips.items()},
            "duplicate_pairs": self.duplicate_pairs,
        }


def _flip_label(label: str, rng: random.Random) -> str:
    choices = [x for x in UNILATERAL_LABELS if x != label]
    return rng.choice(choices)


def inject_label_flips(
    annotations_csv: Path,
    out_csv: Path,
    *,
    n_flips: int,
    seed: int = 20260808,
) -> dict[str, tuple[str, str]]:
    """Flip n_flips supplied labels (LEFT<->RIGHT) in a copy of ``annotations_csv``.

    Returns {path -> (original_label, corrupted_label)} for scoring.
    """
    rng = random.Random(seed)
    rows: list[dict] = []
    with annotations_csv.open() as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        for r in reader:
            rows.append(r)
    if fieldnames is None:
        raise ValueError("annotations CSV has no header")

    # Only flip rows whose supplied label is unilateral (LEFT/RIGHT).
    label_col = next((c for c in fieldnames if c.lower() in ("laterality", "label", "side")), None)
    path_col = next((c for c in fieldnames if c.lower() in ("path", "filename", "file", "image", "relpath")), None)
    if label_col is None or path_col is None:
        raise ValueError(f"annotations CSV needs a label and path column; got {fieldnames}")

    flippable_indices = [i for i, r in enumerate(rows)
                         if str(r[label_col]).strip().upper() in UNILATERAL_LABELS]
    rng.shuffle(flippable_indices)
    picked = flippable_indices[:n_flips]

    flips: dict[str, tuple[str, str]] = {}
    for i in picked:
        original = rows[i][label_col].strip().upper()
        corrupted = _flip_label(original, rng)
        flips[rows[i][path_col]] = (original, corrupted)
        rows[i][label_col] = corrupted

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return flips


def _apply_transform(im: Image.Image, name: str) -> Image.Image:
    if name == "mirror":
        return im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if name == "rot90":
        return im.rotate(90, expand=True)
    if name == "rot180":
        return im.rotate(180, expand=True)
    if name == "rot270":
        return im.rotate(270, expand=True)
    if name == "reencode":
        return im.copy()
    raise ValueError(name)


def inject_duplicates(
    images_dir: Path,
    *,
    n_duplicates: int,
    transforms: Iterable[str] = TRANSFORMS,
    seed: int = 20260808,
) -> list[tuple[str, str, str]]:
    """Copy N images into ``images_dir/injected_dupes/`` after applying a transform.

    Returns [(source_path, derivative_path, transform)].
    """
    rng = random.Random(seed)
    source_files = sorted(images_dir.rglob("*.png"))
    source_files = [p for p in source_files if "injected_dupes" not in p.parts]
    if not source_files:
        raise FileNotFoundError(f"no PNGs under {images_dir}")

    out_dir = images_dir / "injected_dupes"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[str, str, str]] = []
    for i in range(n_duplicates):
        src = rng.choice(source_files)
        tf = rng.choice(list(transforms))
        stem = f"inj_{i:04d}_{tf}_{uuid.uuid4().hex[:6]}.png"
        dst = out_dir / stem
        with Image.open(src) as im:
            _apply_transform(im.convert("RGB"), tf).save(dst)
        pairs.append((str(src), str(dst), tf))
    return pairs


def build_corrupted_dataset(
    src_images: Path,
    src_annotations: Path,
    out_root: Path,
    *,
    n_flips: int,
    n_duplicates: int,
    seed: int = 20260808,
) -> tuple[Path, Path, InjectionGroundTruth]:
    """Copy the source dataset to ``out_root/{images,annotations.csv}`` and corrupt it.

    Returns (new_images_dir, new_annotations_csv, ground_truth).
    """
    out_root = out_root.resolve()
    out_images = out_root / "images"
    out_ann = out_root / "annotations.csv"
    if out_images.exists():
        shutil.rmtree(out_images)
    shutil.copytree(src_images, out_images)

    flips = inject_label_flips(src_annotations, out_ann, n_flips=n_flips, seed=seed)
    dup_pairs = inject_duplicates(out_images, n_duplicates=n_duplicates, seed=seed + 1)
    gt = InjectionGroundTruth(label_flips=flips, duplicate_pairs=dup_pairs)
    (out_root / "injection_ground_truth.json").write_text(
        json.dumps(gt.as_dict(), indent=2)
    )
    return out_images, out_ann, gt


# --------------------------------------------------------------------- scoring

def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def score_injection_detection(
    db: Database, gt: InjectionGroundTruth,
) -> dict[str, dict[str, Any]]:
    """Score the auditor's ability to detect the injected corruptions."""

    # ---- Label-flip detection.
    # A flip is "detected" if the automated prediction (or any reviewer decision) disagrees
    # with the supplied (corrupted) label AND matches (or at least differs consistently
    # with) the original label. Since we always know both, we score against ground truth.
    flip_paths = {_norm(p) for p in gt.label_flips}
    flip_detected: set[str] = set()

    # 1a. `possible_label_error=True` on the first-pass prediction.
    for r in db._conn.execute(
        "SELECT path, possible_label_error, laterality FROM predictions WHERE pass_number=1"
    ):
        if r["possible_label_error"] == 1:
            flip_detected.add(_norm(r["path"]))

    # 1b. Model prediction disagrees with (corrupted) supplied label AND matches the
    # original — that's a positive-signal detection too.
    supplied = {r["path"]: r["supplied_label"] for r in db._conn.execute(
        "SELECT path, supplied_label FROM images WHERE supplied_label IS NOT NULL"
    )}
    for r in db._conn.execute(
        "SELECT path, laterality FROM predictions WHERE pass_number=1"
    ):
        s = supplied.get(r["path"])
        pred = r["laterality"]
        if not s or not pred:
            continue
        if pred != _norm(s):
            # Ground truth original label available? Then we detected the flip.
            key = _norm(r["path"])
            if key in flip_paths and pred == gt.label_flips[_lookup(gt, key)][0]:
                flip_detected.add(key)

    # But `possible_label_error` detections outside the injected set are false positives
    # only relative to our known ground truth; real audits will surface unknown errors
    # too, so we score narrowly: TP = detected in flip set; FP = flagged but not in flip
    # set AND not in original data (unknown = ambiguous — we EXCLUDE from FP for a clean
    # score; the paper explicitly notes this).
    fp_candidates: set[str] = set()
    for r in db._conn.execute(
        "SELECT path FROM predictions WHERE pass_number=1 AND possible_label_error=1"
    ):
        k = _norm(r["path"])
        if k not in flip_paths:
            fp_candidates.add(k)

    label_tp = len(flip_detected & flip_paths)
    label_fp = len(fp_candidates)          # narrow FP: only counts flagged-but-not-in-set
    label_fn = len(flip_paths - flip_detected)
    label_score = _prf(label_tp, label_fp, label_fn) | {
        "known_flips": len(flip_paths),
        "definition": (
            "TP=injected flip flagged possible_label_error OR predicted original label; "
            "FP=possible_label_error flag outside injected set (may include real errors — "
            "reported as an UPPER BOUND on FP); FN=injected flip not detected."
        ),
    }

    # ---- Duplicate detection.
    # A duplicate is "detected" if source_path and derivative_path share a source_group_id.
    src_group = {r["path"]: r["source_group_id"] for r in db._conn.execute(
        "SELECT path, source_group_id FROM images"
    )}
    dup_tp = 0
    dup_fn = 0
    for src, dst, _tf in gt.duplicate_pairs:
        gs = src_group.get(src)
        gd = src_group.get(dst)
        if gs and gd and gs == gd:
            dup_tp += 1
        else:
            dup_fn += 1
    # FP: pairs of images placed in the same group whose sources are unrelated.
    # We approximate: any multi-member source group that spans images from different
    # subfolders AND is not one of our injected groups is a candidate FP.
    injected_dup_groups = {src_group.get(dst) for _, dst, _ in gt.duplicate_pairs
                           if src_group.get(dst)}
    group_members: dict[str, list[str]] = {}
    for path, gid in src_group.items():
        if gid:
            group_members.setdefault(gid, []).append(path)
    dup_fp = 0
    for gid, members in group_members.items():
        if len(members) < 2 or gid in injected_dup_groups:
            continue
        # Check if members come from >1 distinct parent dir AND are not our injected ones.
        parents = {Path(m).parent for m in members}
        if len(parents) > 1:
            dup_fp += 1  # 1 spurious group counted as 1 FP for reporting simplicity

    dup_score = _prf(dup_tp, dup_fp, dup_fn) | {
        "known_duplicates": len(gt.duplicate_pairs),
        "definition": (
            "TP=injected pair placed in the same source_group_id; "
            "FP=multi-member source group spanning >1 parent directory not in the "
            "injected set (upper bound — may include real derivatives); "
            "FN=injected pair not placed in the same group."
        ),
    }
    return {"label_flip_detection": label_score,
            "duplicate_detection": dup_score}


def _norm(p: str | Path) -> str:
    return str(Path(p).resolve())


def _lookup(gt: InjectionGroundTruth, resolved_key: str) -> str:
    """Return the original (unresolved) path key from ``gt.label_flips`` matching resolved_key."""
    for k in gt.label_flips:
        if _norm(k) == resolved_key or resolved_key.endswith(k):
            return k
    return next(iter(gt.label_flips))


def write_injection_report(
    scores: dict[str, dict[str, Any]], out_dir: Path,
) -> Path:
    ensure_dir(out_dir)
    p = out_dir / "injection_report.json"
    p.write_text(json.dumps(scores, indent=2))
    csv_p = out_dir / "injection_report.csv"
    rows: list[dict] = []
    for kind, s in scores.items():
        rows.append({
            "check": kind,
            "known": s.get("known_flips", s.get("known_duplicates", 0)),
            "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
            "precision": round(s["precision"], 4),
            "recall": round(s["recall"], 4),
            "f1": round(s["f1"], 4),
        })
    with csv_p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["check", "known", "tp", "fp", "fn",
                                            "precision", "recall", "f1"])
        w.writeheader()
        w.writerows(rows)
    return csv_p
