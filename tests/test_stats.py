from __future__ import annotations

import pandas as pd

from dermlat_audit.stats import (
    abstention_metrics,
    accuracy_on_answerable_unilateral,
    answerable_split,
    clustered_bootstrap_ci,
    cohens_kappa,
    label_distribution,
    selective_metrics,
    supplied_label_agreement,
)


def test_label_distribution_denominators():
    labels = ["LEFT", "LEFT", "RIGHT", "UNANSWERABLE"]
    d = label_distribution(labels)
    assert d["LEFT"]["count"] == 2
    assert d["LEFT"]["denominator"] == 4
    assert d["LEFT"]["rate"] == 0.5
    assert d["EXCLUDE_CORRUPT_OR_NONCLINICAL"]["count"] == 0


def test_answerable_split():
    labels = ["LEFT", "RIGHT", "BILATERAL_OR_MIDLINE", "UNANSWERABLE",
              "EXCLUDE_CORRUPT_OR_NONCLINICAL"]
    r = answerable_split(labels)
    assert r["total"] == 5
    assert r["answerable"]["count"] == 3
    assert r["unanswerable"]["count"] == 1
    assert r["excluded"]["count"] == 1


def test_supplied_label_agreement_and_none_handling():
    pred = ["LEFT", "RIGHT", "LEFT", "UNANSWERABLE"]
    supp = ["LEFT", "LEFT", None, "UNANSWERABLE"]
    r = supplied_label_agreement(pred, supp)
    # None is excluded; comparable set is 3, agreements are 2 (LEFT/LEFT + UNANS/UNANS).
    assert r.n_comparable == 3
    assert r.n_agree == 2


def test_accuracy_on_answerable_unilateral_ignores_non_unilateral():
    pred = ["LEFT", "RIGHT", "LEFT", "UNANSWERABLE"]
    supp = ["LEFT", "LEFT", "BILATERAL_OR_MIDLINE", "UNANSWERABLE"]
    r = accuracy_on_answerable_unilateral(pred, supp)
    # Only rows 0 (correct) and 1 (wrong) count. Accuracy = 0.5.
    assert r["n"] == 2
    assert r["accuracy"] == 0.5


def test_selective_metrics_coverage():
    pred = ["LEFT", "LEFT", "UNANSWERABLE", "UNANSWERABLE"]
    supp = ["LEFT", "RIGHT", "LEFT", "LEFT"]
    r = selective_metrics(pred, supp)
    assert r["coverage"] == 0.5
    # selective_n counts non-abstained rows whose supplied label is unilateral: 2. correct: 1.
    assert r["selective_n"] == 2
    assert r["selective_correct"] == 1
    assert r["selective_accuracy"] == 0.5


def test_abstention_metrics_sens_spec():
    pred = ["UNANSWERABLE", "UNANSWERABLE", "LEFT", "LEFT"]
    supp = ["UNANSWERABLE", "LEFT", "LEFT", "RIGHT"]
    r = abstention_metrics(pred, supp)
    # Supplied UNANS: 1  → abstained 1 → sens = 1.0
    # Supplied unilat: 3 → answered 2   → spec = 2/3
    assert abs(r["sensitivity"] - 1.0) < 1e-9
    assert abs(r["specificity"] - 2 / 3) < 1e-9


def test_cohens_kappa_perfect_agreement_is_one():
    a = ["LEFT", "RIGHT", "LEFT"]
    r = cohens_kappa(a, a)
    assert abs(r["kappa"] - 1.0) < 1e-9


def test_cohens_kappa_no_agreement():
    a = ["LEFT", "LEFT", "LEFT"]
    b = ["RIGHT", "RIGHT", "RIGHT"]
    r = cohens_kappa(a, b)
    assert r["kappa"] <= 0


def test_clustered_bootstrap_ci_returns_interval():
    df = pd.DataFrame({
        "path": [f"/{i}.png" for i in range(20)],
        "src": [f"g{i % 5}" for i in range(20)],
        "correct": [1] * 15 + [0] * 5,
    })
    r = clustered_bootstrap_ci(
        lambda d: d["correct"].mean(),
        df, cluster_col="src", iterations=200, seed=1,
    )
    assert 0 <= r["ci_low"] <= r["point"] <= r["ci_high"] <= 1
