"""Paper-ready figures + calibration metrics.

Produces the extra artefacts the IEEE URTC draft needs:
  fig_2_confusion_matrix.png     — supplied vs. final label
  fig_3_reliability_diagram.png  — accuracy per confidence bin + ECE annotation
  fig_4_coverage_accuracy.png    — selective-classification curve
  fig_5_source_group_sizes.png   — histogram of source-group member counts

Also returns calibration metrics (Expected Calibration Error, MCE, Brier-like
top-label score) that go straight into metrics.json and the LaTeX numbers file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import UNILATERAL_LABELS
from .db import Database
from .utils import ensure_dir

# --------------------------------------------------------------------- data

def _paired_pred_supplied(db: Database) -> pd.DataFrame:
    """Rows with a first-pass prediction AND a supplied unilateral label."""
    rows = list(db._conn.execute(
        """
        SELECT i.path, i.supplied_label,
               p.laterality AS pred, p.confidence AS conf, p.answerable
        FROM images i
        LEFT JOIN predictions p ON p.path = i.path AND p.pass_number = 1
        WHERE i.readable = 1
        """
    ))
    if not rows:
        return pd.DataFrame(columns=["path", "supplied_label", "pred", "conf", "answerable"])
    df = pd.DataFrame([dict(r) for r in rows])
    df["supplied_norm"] = df["supplied_label"].astype(str).str.upper().str.replace(" ", "_")
    return df


# --------------------------------------------------------------------- calibration

def calibration_metrics(
    confidences: list[int], correctness: list[bool], n_bins: int = 10,
) -> dict[str, Any]:
    """Expected Calibration Error, Max Calibration Error, and top-label Brier score.

    Confidences are integers 0-100. Correctness is bool per prediction.
    """
    if not confidences:
        return {"ece": None, "mce": None, "brier_top": None, "n": 0, "bins": []}
    conf = np.asarray(confidences, dtype=float) / 100.0
    correct = np.asarray(correctness, dtype=float)
    n = len(conf)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bins: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        # Include right edge on the last bin.
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        m = int(mask.sum())
        if m == 0:
            bins.append({"bin_low": float(lo), "bin_high": float(hi),
                         "n": 0, "acc": None, "conf_mean": None})
            continue
        acc = float(correct[mask].mean())
        conf_mean = float(conf[mask].mean())
        weight = m / n
        gap = abs(acc - conf_mean)
        ece += weight * gap
        mce = max(mce, gap)
        bins.append({"bin_low": float(lo), "bin_high": float(hi),
                     "n": m, "acc": acc, "conf_mean": conf_mean})
    brier_top = float(((conf - correct) ** 2).mean())
    return {"ece": ece, "mce": mce, "brier_top": brier_top, "n": n, "bins": bins}


def coverage_accuracy_curve(
    confidences: list[int], correctness: list[bool],
) -> list[dict[str, float]]:
    """Selective-classification curve: sort by confidence desc, plot cumulative accuracy."""
    if not confidences:
        return []
    order = np.argsort(-np.asarray(confidences))
    correct = np.asarray(correctness, dtype=float)[order]
    n = len(correct)
    cum_correct = np.cumsum(correct)
    coverage = (np.arange(1, n + 1)) / n
    accuracy = cum_correct / np.arange(1, n + 1)
    return [{"coverage": float(c), "accuracy": float(a)}
            for c, a in zip(coverage, accuracy, strict=True)]


# --------------------------------------------------------------------- figures

def render_all(db: Database, metrics: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Extend metrics with calibration; emit paper-ready figures.

    Returns extra metrics to be merged into the top-level metrics.json.
    """
    figures_dir = ensure_dir(out_dir / "figures")
    tables_dir = ensure_dir(out_dir / "tables")

    df = _paired_pred_supplied(db)
    df_unilat = df[df["supplied_norm"].isin(UNILATERAL_LABELS)
                   & df["conf"].notnull() & df["pred"].notnull()].copy()

    confidences = df_unilat["conf"].astype(int).tolist()
    correctness = (df_unilat["pred"] == df_unilat["supplied_norm"]).tolist()

    calib = calibration_metrics(confidences, correctness, n_bins=10)
    curve = coverage_accuracy_curve(confidences, correctness)

    # Table: calibration bins
    if calib["bins"]:
        pd.DataFrame(calib["bins"]).to_csv(
            tables_dir / "table_4_calibration_bins.csv", index=False
        )
    # Table: coverage-accuracy curve
    if curve:
        pd.DataFrame(curve).to_csv(
            tables_dir / "table_5_coverage_accuracy.csv", index=False
        )

    # Source group size histogram
    sg = list(db._conn.execute(
        "SELECT member_count FROM source_groups"
    ))
    sg_counts = [r["member_count"] for r in sg]
    if sg_counts:
        pd.DataFrame({"member_count": sg_counts}).to_csv(
            tables_dir / "table_6_source_group_sizes.csv", index=False
        )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"calibration": calib, "coverage_accuracy_curve": curve}

    # Fig 2: confusion matrix (supplied vs. final).
    cm = metrics.get("supplied_label_agreement", {}).get("confusion_matrix", {})
    if cm:
        labels = sorted(cm.keys())
        m = np.array([[cm[r].get(c, 0) for c in labels] for r in labels], dtype=float)
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        im = ax.imshow(m, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("supplied")
        ax.set_title("Confusion matrix (supplied vs. final)")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{int(m[i, j])}",
                        ha="center", va="center",
                        color=("white" if m[i, j] > m.max() / 2 else "black"),
                        fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.savefig(figures_dir / "fig_2_confusion_matrix.png", dpi=150)
        plt.close(fig)

    # Fig 3: reliability diagram
    if calib["bins"] and any(b["acc"] is not None for b in calib["bins"]):
        fig, ax = plt.subplots(figsize=(4.0, 3.6))
        centers = [(b["bin_low"] + b["bin_high"]) / 2 for b in calib["bins"]]
        accs = [b["acc"] if b["acc"] is not None else 0.0 for b in calib["bins"]]
        weights = [b["n"] for b in calib["bins"]]
        ax.bar(centers, accs, width=0.09, alpha=0.7, edgecolor="black", label="accuracy")
        ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
        ax.set_xlabel("confidence bin")
        ax.set_ylabel("accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Reliability diagram (ECE={calib['ece']:.3f})")
        ax.legend(fontsize=8, loc="lower right")
        # secondary axis with bin counts
        ax2 = ax.twinx()
        ax2.plot(centers, weights, "o-", color="C1", alpha=0.6, label="n")
        ax2.set_ylabel("n per bin", color="C1")
        plt.tight_layout()
        plt.savefig(figures_dir / "fig_3_reliability_diagram.png", dpi=150)
        plt.close(fig)

    # Fig 4: coverage-accuracy curve
    if curve:
        fig, ax = plt.subplots(figsize=(4.0, 3.4))
        cov = [x["coverage"] for x in curve]
        acc = [x["accuracy"] for x in curve]
        ax.plot(cov, acc)
        ax.set_xlabel("coverage (fraction of images accepted, sorted by conf.)")
        ax.set_ylabel("selective accuracy")
        ax.set_title("Coverage-accuracy curve")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "fig_4_coverage_accuracy.png", dpi=150)
        plt.close(fig)

    # Fig 5: source-group size histogram
    if sg_counts:
        fig, ax = plt.subplots(figsize=(4.0, 3.2))
        max_size = max(sg_counts)
        bins = (np.arange(1, max_size + 2) - 0.5).tolist()
        ax.hist(sg_counts, bins=bins, edgecolor="black")
        ax.set_xlabel("group size (images per source_group_id)")
        ax.set_ylabel("count of groups")
        ax.set_yscale("log")
        ax.set_title("Source-group size distribution")
        plt.tight_layout()
        plt.savefig(figures_dir / "fig_5_source_group_sizes.png", dpi=150)
        plt.close(fig)

    return {"calibration": calib, "coverage_accuracy_curve_summary": {
        "n_points": len(curve),
        "accuracy_at_50pct_coverage": (
            curve[max(0, len(curve) // 2 - 1)]["accuracy"] if curve else None
        ),
        "accuracy_at_full_coverage": curve[-1]["accuracy"] if curve else None,
    }}
