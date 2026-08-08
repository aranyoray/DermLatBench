"""All statistical analyses required by the audit.

Every function that returns a rate also returns the exact numerator and denominator
so nothing is quoted without provenance.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

from .constants import (
    ABSTENTION_LABELS,
    ALL_LABELS,
    LABEL_EXCLUDE,
    LABEL_UNANSWERABLE,
    UNILATERAL_LABELS,
)

# --------------------------------------------------------------------- distributions

def label_distribution(labels: Iterable[str]) -> dict[str, dict[str, Any]]:
    labels = list(labels)
    total = len(labels)
    counts = Counter(labels)
    out: dict[str, dict[str, Any]] = {}
    for label in ALL_LABELS:
        n = int(counts.get(label, 0))
        out[label] = {"count": n, "denominator": total,
                      "rate": (n / total) if total else 0.0}
    return out


def answerable_split(labels: Iterable[str]) -> dict[str, Any]:
    labels = list(labels)
    total = len(labels)
    answerable = sum(1 for x in labels if x in UNILATERAL_LABELS or x == "BILATERAL_OR_MIDLINE")
    unanswerable = sum(1 for x in labels if x == LABEL_UNANSWERABLE)
    excluded = sum(1 for x in labels if x == LABEL_EXCLUDE)
    return {
        "total": total,
        "answerable": {"count": answerable, "rate": answerable / total if total else 0.0},
        "unanswerable": {"count": unanswerable, "rate": unanswerable / total if total else 0.0},
        "excluded": {"count": excluded, "rate": excluded / total if total else 0.0},
    }


# --------------------------------------------------------------------- agreement / accuracy

@dataclass
class AgreementResult:
    n_comparable: int
    n_agree: int
    agreement: float
    confusion_matrix: dict[str, dict[str, int]]
    by_supplied_label: dict[str, dict[str, Any]] = field(default_factory=dict)


def supplied_label_agreement(
    predicted: Sequence[str], supplied: Sequence[str | None],
) -> AgreementResult:
    """Compare a predicted label vs a supplied (existing) label for the same image.

    Rows with no supplied label are excluded from the denominator (and reported separately
    at the call site).
    """
    if len(predicted) != len(supplied):
        raise ValueError("length mismatch")
    pairs = [(p, s) for p, s in zip(predicted, supplied, strict=True) if s is not None]
    n = len(pairs)
    if n == 0:
        return AgreementResult(0, 0, 0.0, {})
    agree = sum(1 for p, s in pairs if _normalize(p) == _normalize(s))
    y_pred = [_normalize(p) for p, _ in pairs]
    y_true = [_normalize(s) for _, s in pairs]
    labels = sorted(set(y_pred) | set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_dict = {row_label: {col_label: int(cm[i][j]) for j, col_label in enumerate(labels)}
               for i, row_label in enumerate(labels)}
    by_label: dict[str, dict[str, Any]] = {}
    for label in labels:
        denom = sum(1 for s in y_true if s == label)
        num = sum(1 for p, s in zip(y_pred, y_true, strict=True) if s == label and p == s)
        by_label[label] = {"n": denom, "agree": num,
                           "recall": (num / denom) if denom else 0.0}
    return AgreementResult(n, agree, agree / n, cm_dict, by_label)


def _normalize(label: str | None) -> str:
    if label is None:
        return ""
    return str(label).strip().upper().replace(" ", "_")


def accuracy_on_answerable_unilateral(
    predicted: Sequence[str], supplied: Sequence[str | None],
) -> dict[str, Any]:
    """Accuracy considering only rows whose SUPPLIED label is LEFT or RIGHT."""
    pairs = [(p, s) for p, s in zip(predicted, supplied, strict=True)
             if s is not None and _normalize(s) in UNILATERAL_LABELS]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "correct": 0, "accuracy": 0.0, "macro_recall": 0.0,
                "balanced_accuracy": 0.0}
    y_pred = [_normalize(p) for p, _ in pairs]
    y_true = [_normalize(s) for _, s in pairs]
    correct = sum(1 for a, b in zip(y_pred, y_true, strict=True) if a == b)
    macro_recall = float(recall_score(
        y_true, y_pred, labels=list(UNILATERAL_LABELS), average="macro", zero_division=0
    ))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    return {"n": n, "correct": correct, "accuracy": correct / n,
            "macro_recall": macro_recall, "balanced_accuracy": bal_acc}


# --------------------------------------------------------------------- selective / abstention

def selective_metrics(
    predicted: Sequence[str], supplied: Sequence[str | None],
) -> dict[str, Any]:
    """Coverage (fraction not abstained) and selective accuracy on unilateral supplied rows."""
    total = len(predicted)
    non_abstained_mask = [p not in ABSTENTION_LABELS for p in predicted]
    coverage = sum(non_abstained_mask) / total if total else 0.0

    pairs = [(p, s) for p, s in zip(predicted, supplied, strict=True)
             if s is not None and _normalize(s) in UNILATERAL_LABELS
             and p not in ABSTENTION_LABELS]
    n = len(pairs)
    correct = sum(1 for p, s in pairs if _normalize(p) == _normalize(s))
    return {"coverage": coverage, "selective_n": n,
            "selective_correct": correct,
            "selective_accuracy": (correct / n) if n else 0.0}


def abstention_metrics(
    predicted: Sequence[str], supplied: Sequence[str | None],
) -> dict[str, Any]:
    """Treat UNANSWERABLE (predicted) as an abstention.

    Sensitivity = P(predict abstain | supplied is UNANSWERABLE)  — abstain-when-unclear
    Specificity = P(predict answer | supplied is unilateral)     — answer-when-clear
    """
    unans_supplied = [(_normalize(s) == LABEL_UNANSWERABLE) for s in supplied]
    unilat_supplied = [(_normalize(s) in UNILATERAL_LABELS) for s in supplied]
    abstained = [(p == LABEL_UNANSWERABLE) for p in predicted]
    n_unans = sum(unans_supplied)
    n_unilat = sum(unilat_supplied)
    if n_unans == 0 and n_unilat == 0:
        return {"note": "no supplied labels of interest",
                "sensitivity": None, "specificity": None}
    tp = sum(1 for a, u in zip(abstained, unans_supplied, strict=True) if a and u)
    tn = sum(1 for a, u in zip(abstained, unilat_supplied, strict=True) if (not a) and u)
    sens = (tp / n_unans) if n_unans else None
    spec = (tn / n_unilat) if n_unilat else None
    return {"n_supplied_unanswerable": n_unans, "n_supplied_unilateral": n_unilat,
            "sensitivity": sens, "specificity": spec}


# --------------------------------------------------------------------- reviewer agreement

def cohens_kappa(rater_a: Sequence[str], rater_b: Sequence[str]) -> dict[str, Any]:
    """Cohen's kappa on paired categorical labels."""
    if len(rater_a) != len(rater_b):
        raise ValueError("length mismatch")
    if not rater_a:
        return {"n": 0, "kappa": None}
    labels = sorted(set(rater_a) | set(rater_b))
    m = confusion_matrix(rater_a, rater_b, labels=labels)
    n = m.sum()
    po = float(np.trace(m)) / n
    row_marginals = m.sum(axis=1) / n
    col_marginals = m.sum(axis=0) / n
    pe = float((row_marginals * col_marginals).sum())
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 1.0
    return {"n": int(n), "po": po, "pe": pe, "kappa": kappa}


def krippendorff_alpha(reviewers: dict[str, Sequence[str | None]]) -> dict[str, Any]:
    """Krippendorff's α across N reviewers. Requires ``krippendorff`` package."""
    try:
        import krippendorff
    except ImportError:
        return {"alpha": None, "note": "install with pip install 'dermlat-audit[stats]'"}
    values = list(reviewers.values())
    n_units = max(len(v) for v in values)
    label_to_int: dict[str, int] = {}
    def enc(x: str | None) -> Any:
        if x is None:
            return np.nan
        if x not in label_to_int:
            label_to_int[x] = len(label_to_int) + 1
        return label_to_int[x]

    matrix = np.array([
        [enc(v[i]) if i < len(v) else np.nan for i in range(n_units)]
        for v in values
    ], dtype=float)
    try:
        alpha = float(krippendorff.alpha(reliability_data=matrix, level_of_measurement="nominal"))
    except Exception as exc:
        return {"alpha": None, "note": f"krippendorff failed: {exc}"}
    return {"alpha": alpha, "n_units": n_units, "n_reviewers": len(reviewers)}


# --------------------------------------------------------------------- clustered bootstrap CI

def clustered_bootstrap_ci(
    metric_fn,
    df: pd.DataFrame,
    cluster_col: str,
    *,
    iterations: int = 2000,
    ci: float = 0.95,
    seed: int = 20260808,
) -> dict[str, float]:
    """95% clustered bootstrap CI for a scalar metric_fn(df) -> float.

    Clusters are ``source_group_id`` — we resample clusters (with replacement), not rows.
    """
    if df.empty:
        return {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0, "iterations": 0}
    point = float(metric_fn(df))
    rng = random.Random(seed)
    clusters = df.groupby(cluster_col).indices  # {cluster -> np.array(indices)}
    keys = list(clusters.keys())
    values = []
    for _ in range(iterations):
        sampled = rng.choices(keys, k=len(keys))
        idxs = np.concatenate([clusters[k] for k in sampled])
        sample = df.iloc[idxs]
        try:
            values.append(float(metric_fn(sample)))
        except Exception:
            continue
    if not values:
        return {"point": point, "ci_low": point, "ci_high": point, "iterations": 0}
    lo = float(np.quantile(values, (1 - ci) / 2))
    hi = float(np.quantile(values, 1 - (1 - ci) / 2))
    return {"point": point, "ci_low": lo, "ci_high": hi, "iterations": len(values)}


# --------------------------------------------------------------------- confidence-stratified

def confidence_stratified(
    predicted: Sequence[str], supplied: Sequence[str | None], confidences: Sequence[int | None],
    bins: Sequence[tuple[int, int]] = ((0, 40), (40, 60), (60, 80), (80, 101)),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for lo, hi in bins:
        idx = [i for i, c in enumerate(confidences) if c is not None and lo <= c < hi]
        if not idx:
            out.append({"bin": f"{lo}-{hi-1}", "n": 0, "accuracy": None})
            continue
        pair = [(predicted[i], supplied[i]) for i in idx
                if supplied[i] is not None and _normalize(supplied[i]) in UNILATERAL_LABELS]
        n = len(pair)
        correct = sum(1 for p, s in pair if _normalize(p) == _normalize(s))
        out.append({
            "bin": f"{lo}-{hi-1}", "n": n,
            "accuracy": (correct / n) if n else None,
            "coverage_of_bin": len(idx),
        })
    return out


# --------------------------------------------------------------------- by-source / by-site

def group_by(df: pd.DataFrame, col: str, metric_col: str) -> list[dict[str, Any]]:
    """Simple per-group counts. Doesn't compute agreement — just distribution per group."""
    if col not in df.columns:
        return []
    out = []
    for value, sub in df.groupby(col, dropna=False):
        counts = Counter(sub[metric_col].dropna())
        out.append({str(col): (str(value) if not pd.isna(value) else None),
                    "n": len(sub),
                    "label_counts": {k: int(v) for k, v in counts.items()}})
    return out
