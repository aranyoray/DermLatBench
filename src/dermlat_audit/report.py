"""Build all output artefacts (CSVs, metrics.json, Markdown report, run manifest)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import PIPELINE_VERSION, __version__
from .config import AppConfig
from .db import Database
from .paper_figures import render_all as render_paper_figures
from .splits import assignments_to_dataframe, make_splits, validate_no_leakage
from .stats import (
    abstention_metrics,
    accuracy_on_answerable_unilateral,
    answerable_split,
    clustered_bootstrap_ci,
    cohens_kappa,
    confidence_stratified,
    group_by,
    krippendorff_alpha,
    label_distribution,
    selective_metrics,
    supplied_label_agreement,
)
from .utils import ensure_dir, utc_now_iso
from .verify_claims import overall_publication_ready, verify_claims


def _rows_to_df(rows) -> pd.DataFrame:
    return pd.DataFrame([dict(r) for r in rows])


def _combined_labels(db: Database) -> pd.DataFrame:
    """One row per image with best-available laterality label:
        adjudicated > review_2 > review_1 > model pass_1
    """
    images = _rows_to_df(db.all_images())
    preds1 = _rows_to_df(db._conn.execute(
        "SELECT path, laterality, confidence, answerable FROM predictions WHERE pass_number=1"
    ).fetchall())
    r1 = _rows_to_df(db.reviews_by_reviewer("reviewer_1"))
    r2 = _rows_to_df(db.reviews_by_reviewer("reviewer_2"))
    adj = _rows_to_df(db.all_adjudications())
    if preds1.empty:
        preds1 = pd.DataFrame(columns=["path", "laterality", "confidence", "answerable"])
    m = images[["path", "supplied_label", "source_group_id"]].copy()
    m = m.merge(preds1.rename(columns={"laterality": "model_laterality",
                                       "confidence": "model_confidence",
                                       "answerable": "model_answerable"}),
                on="path", how="left")
    m = m.merge(r1.rename(columns={"laterality": "r1_laterality"})[["path", "r1_laterality"]],
                on="path", how="left") if not r1.empty else m
    m = m.merge(r2.rename(columns={"laterality": "r2_laterality"})[["path", "r2_laterality"]],
                on="path", how="left") if not r2.empty else m
    m = m.merge(adj.rename(columns={"laterality": "adjudicated_laterality"})[["path", "adjudicated_laterality"]],
                on="path", how="left") if not adj.empty else m
    for col in ("r1_laterality", "r2_laterality", "adjudicated_laterality"):
        if col not in m.columns:
            m[col] = None
    m["final_laterality"] = m["adjudicated_laterality"].fillna(m["r2_laterality"]).fillna(m["r1_laterality"]).fillna(m["model_laterality"])
    return m


def write_csvs(db: Database, out_dir: Path) -> dict[str, Path]:
    ensure_dir(out_dir)
    out: dict[str, Path] = {}

    manifest = _rows_to_df(db.all_images())
    (out_dir / "complete_manifest.csv").write_text(manifest.to_csv(index=False))
    out["complete_manifest.csv"] = out_dir / "complete_manifest.csv"

    preds = _rows_to_df(db.all_predictions())
    (out_dir / "automated_predictions.csv").write_text(preds.to_csv(index=False))
    out["automated_predictions.csv"] = out_dir / "automated_predictions.csv"

    rq = _rows_to_df(db.review_queue())
    (out_dir / "flagged_human_review.csv").write_text(rq.to_csv(index=False))
    out["flagged_human_review.csv"] = out_dir / "flagged_human_review.csv"

    r1 = _rows_to_df(db.reviews_by_reviewer("reviewer_1"))
    (out_dir / "reviewer_1.csv").write_text(r1.to_csv(index=False))
    out["reviewer_1.csv"] = out_dir / "reviewer_1.csv"

    r2 = _rows_to_df(db.reviews_by_reviewer("reviewer_2"))
    (out_dir / "reviewer_2.csv").write_text(r2.to_csv(index=False))
    out["reviewer_2.csv"] = out_dir / "reviewer_2.csv"

    adj = _rows_to_df(db.all_adjudications())
    (out_dir / "adjudicated_labels.csv").write_text(adj.to_csv(index=False))
    out["adjudicated_labels.csv"] = out_dir / "adjudicated_labels.csv"

    dup_groups = _rows_to_df(db._conn.execute("SELECT * FROM duplicate_groups").fetchall())
    (out_dir / "duplicate_groups.csv").write_text(dup_groups.to_csv(index=False))
    out["duplicate_groups.csv"] = out_dir / "duplicate_groups.csv"

    src_groups = _rows_to_df(db._conn.execute("SELECT * FROM source_groups").fetchall())
    (out_dir / "source_groups.csv").write_text(src_groups.to_csv(index=False))
    out["source_groups.csv"] = out_dir / "source_groups.csv"

    return out


def build_splits(db: Database, cfg: AppConfig, out_dir: Path) -> tuple[pd.DataFrame, bool, str]:
    rows = [dict(r) for r in db.all_images() if r["readable"]]
    assignments = make_splits(rows, cfg.splits)
    try:
        validate_no_leakage(assignments)
        leak_note = "no leakage detected"
        ok = True
    except Exception as exc:
        leak_note = str(exc)
        ok = False
    df = assignments_to_dataframe(assignments)
    (out_dir / "leakage_safe_splits.csv").write_text(df.to_csv(index=False))
    return df, ok, leak_note


def compute_metrics(db: Database, cfg: AppConfig) -> dict[str, Any]:
    combined = _combined_labels(db)
    # Keep only rows with a final laterality; align supplied + confidence to the same rows.
    aligned = combined.dropna(subset=["final_laterality"]).copy()
    final = aligned["final_laterality"].astype(str).tolist()
    supplied = aligned["supplied_label"].tolist()

    # denominators
    preflight = db.get_meta("preflight") or {}
    total_discovered = int(preflight.get("total_discovered", len(combined)))
    corrupt = int(preflight.get("corrupt", 0))
    unsupported = int(preflight.get("unsupported", 0))

    label_dist = label_distribution(final)
    ans_split = answerable_split(final)
    supplied_norm = [s if s is not None and str(s).strip() != "" else None for s in supplied]
    agreement = supplied_label_agreement(final, supplied_norm)
    acc_unilat = accuracy_on_answerable_unilateral(final, supplied_norm)
    sel = selective_metrics(final, supplied_norm)
    abst = abstention_metrics(final, supplied_norm)

    confidences = aligned["model_confidence"].where(aligned["model_confidence"].notnull(), None).tolist()
    conf_strat = confidence_stratified(final, supplied_norm, [int(c) if c is not None else None for c in confidences])

    # by source group / anatomical site
    site_col = "anatomical_site"
    if site_col not in combined.columns:
        combined[site_col] = None
    by_site = group_by(combined, site_col, "final_laterality")
    by_source_group = group_by(combined, "source_group_id", "final_laterality")

    # Reviewer agreement
    r1 = [dict(r) for r in db.reviews_by_reviewer("reviewer_1")]
    r2 = [dict(r) for r in db.reviews_by_reviewer("reviewer_2")]
    kappa: dict[str, Any] = {"n": 0, "kappa": None}
    alpha: dict[str, Any] = {"alpha": None}
    if r1 and r2:
        paths = sorted({r["path"] for r in r1} & {r["path"] for r in r2})
        d1 = {r["path"]: r["laterality"] for r in r1}
        d2 = {r["path"]: r["laterality"] for r in r2}
        a = [d1[p] for p in paths]
        b = [d2[p] for p in paths]
        kappa = cohens_kappa(a, b)
        alpha = krippendorff_alpha({"r1": a, "r2": b})

    # clustered bootstrap CI on accuracy (per source group), on rows with supplied unilateral labels.
    from .constants import UNILATERAL_LABELS
    boot_df = combined.dropna(subset=["supplied_label"]).copy()
    boot_df["_sup_norm"] = boot_df["supplied_label"].astype(str).str.upper().str.replace(" ", "_")
    boot_df = boot_df[boot_df["_sup_norm"].isin(UNILATERAL_LABELS)].copy()

    def _acc(df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        pred_norm = df["final_laterality"].astype(str).str.upper().str.replace(" ", "_")
        return float((pred_norm == df["_sup_norm"]).mean())

    boot = clustered_bootstrap_ci(
        _acc, boot_df, cluster_col="source_group_id",
        iterations=cfg.stats.bootstrap_iterations, seed=cfg.stats.bootstrap_seed,
    )

    metrics = {
        "generated_at_utc": utc_now_iso(),
        "denominators": {
            "total_discovered": total_discovered,
            "successfully_processed": len(final),
            "corrupt": corrupt,
            "unsupported": unsupported,
            "unprocessed": max(0, total_discovered - len(final) - corrupt - unsupported),
        },
        "label_distribution": label_dist,
        "answerable_split": ans_split,
        "supplied_label_agreement": {
            "n_comparable": agreement.n_comparable,
            "n_agree": agreement.n_agree,
            "agreement": agreement.agreement,
            "confusion_matrix": agreement.confusion_matrix,
            "by_supplied_label": agreement.by_supplied_label,
        },
        "accuracy_on_answerable_unilateral": acc_unilat,
        "selective": sel,
        "abstention": abst,
        "confidence_stratified": conf_strat,
        "by_anatomical_site": by_site,
        "by_source_group": by_source_group,
        "reviewer_agreement": {"cohens_kappa": kappa, "krippendorffs_alpha": alpha},
        "clustered_bootstrap_accuracy_ci": boot,
        "review_queue_size": len(list(db.review_queue())),
    }

    # Verify claims
    claim_results = verify_claims(
        cfg.claims,
        observed_total=total_discovered,
        observed_answerable=ans_split["answerable"]["count"],
        observed_unanswerable=ans_split["unanswerable"]["count"],
        observed_agreement=agreement.agreement if agreement.n_comparable > 0 else None,
    )
    metrics["claim_verification"] = [c.as_dict() for c in claim_results]
    metrics["publication_ready"] = overall_publication_ready(claim_results)
    return metrics


# --------------------------------------------------------------------- markdown / tables

def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _label_dist_table(dist: dict[str, dict[str, Any]]) -> str:
    lines = ["| Label | Count | Rate |", "|---|---:|---:|"]
    for label, v in dist.items():
        lines.append(f"| `{label}` | {v['count']} | {v['rate']:.4f} |")
    return "\n".join(lines)


def _confusion_table(cm: dict[str, dict[str, int]]) -> str:
    if not cm:
        return "*(no comparable rows — no supplied labels found)*"
    labels = sorted(cm.keys())
    header = "| supplied ↓ / predicted → | " + " | ".join(f"`{c}`" for c in labels) + " |"
    align = "|---|" + "|".join(["---:"] * len(labels)) + "|"
    rows = []
    for r in labels:
        cells = [str(cm[r].get(c, 0)) for c in labels]
        rows.append(f"| `{r}` | " + " | ".join(cells) + " |")
    return "\n".join([header, align, *rows])


def _claims_table(claims: list[dict[str, Any]]) -> str:
    lines = ["| Claim | Claimed | Observed | Status | Notes |",
             "|---|---:|---:|:-:|---|"]
    for c in claims:
        lines.append(
            f"| {c['claim']} | {_fmt(c['claim_value'])} | {_fmt(c['observed_value'])} | "
            f"**{c['status']}** | {c['notes']} |"
        )
    return "\n".join(lines)


def _final_statement(metrics: dict[str, Any]) -> str:
    d = metrics["denominators"]
    X = d["successfully_processed"]
    Y = d["total_discovered"]
    Z = d["unprocessed"]
    C = d["corrupt"] + d["unsupported"]
    N = metrics["review_queue_size"]
    ready = "ready" if metrics["publication_ready"] else "not ready"
    return (
        f"**Processed {X:,} of {Y:,} discovered images; unprocessed {Z:,}; "
        f"corrupt/excluded {C:,}; human-review queue {N:,}. "
        f"Dataset-wide claims are [{ready}] for publication.**"
    )


def build_report(cfg: AppConfig, metrics: dict[str, Any], out_dir: Path) -> Path:
    md = out_dir / "audit_report.md"
    lines: list[str] = []
    lines.append("# DermLatBench audit report")
    lines.append("")
    lines.append(f"_Generated_: `{metrics['generated_at_utc']}`")
    lines.append(f"_Pipeline_: `dermlat-audit=={__version__}`, pipeline v`{PIPELINE_VERSION}`")
    lines.append("")
    lines.append("## Methods")
    lines.append(
        "Every image under `paths.images_dir` was discovered, hashed (SHA-256), and its "
        "physical properties (dimensions, format, size, EXIF presence) recorded. Exact "
        "duplicates were grouped by SHA-256. Near-duplicates (mirrors, rotations, crops, "
        "resizes) were grouped by perceptual-hash bucketing plus per-bucket Hamming-distance "
        "comparison over eight canonical orientations, then reduced by union-find into a "
        "single `source_group_id` per equivalence class. Blinded local vision-language "
        "inference used the "
        f"`{cfg.backend}` backend (model `{cfg.mlx.model if cfg.backend == 'mlx' else cfg.ollama.model}`). "
        "The model received only image pixels and a fixed prompt — filenames, folders, EXIF, "
        "and supplied labels were withheld. Responses were parsed and validated against a "
        "fixed JSON schema; invalid or low-confidence responses received a second independent "
        "blinded pass. Disagreements between passes were flagged for human review. Two "
        "reviewers may independently re-label the flagged queue via a local Streamlit UI, "
        "with Reviewer 2 blinded to Reviewer 1's decision until adjudication. "
        "Metrics were computed with exact denominators; clustered bootstrap 95 % CIs use "
        "`source_group_id` as the cluster."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("### Denominators")
    d = metrics["denominators"]
    lines.append(
        f"- total_discovered = **{d['total_discovered']:,}**\n"
        f"- successfully_processed = **{d['successfully_processed']:,}**\n"
        f"- unprocessed = **{d['unprocessed']:,}**\n"
        f"- corrupt = **{d['corrupt']:,}**\n"
        f"- unsupported = **{d['unsupported']:,}**"
    )
    lines.append("")
    lines.append("### Label distribution")
    lines.append(_label_dist_table(metrics["label_distribution"]))
    lines.append("")
    a = metrics["answerable_split"]
    lines.append("### Answerable / Unanswerable / Excluded")
    lines.append(
        f"- answerable: **{a['answerable']['count']:,}** ({a['answerable']['rate']:.4f})\n"
        f"- unanswerable: **{a['unanswerable']['count']:,}** ({a['unanswerable']['rate']:.4f})\n"
        f"- excluded: **{a['excluded']['count']:,}** ({a['excluded']['rate']:.4f})"
    )
    lines.append("")
    ag = metrics["supplied_label_agreement"]
    lines.append("### Supplied-label agreement")
    lines.append(
        f"- n_comparable = **{ag['n_comparable']:,}**\n"
        f"- n_agree = **{ag['n_agree']:,}**\n"
        f"- agreement = **{ag['agreement']:.4f}**"
    )
    lines.append("")
    lines.append("#### Confusion matrix (supplied vs final)")
    lines.append(_confusion_table(ag["confusion_matrix"]))
    lines.append("")
    acc = metrics["accuracy_on_answerable_unilateral"]
    lines.append("### Accuracy on answerable unilateral (supplied LEFT/RIGHT)")
    lines.append(
        f"- n = **{acc['n']:,}**, correct = **{acc['correct']:,}**\n"
        f"- accuracy = **{acc['accuracy']:.4f}**, macro_recall = **{acc['macro_recall']:.4f}**, "
        f"balanced_accuracy = **{acc['balanced_accuracy']:.4f}**"
    )
    lines.append("")
    sel = metrics["selective"]
    lines.append("### Coverage / selective accuracy")
    lines.append(
        f"- coverage = **{sel['coverage']:.4f}**  "
        f"selective_n = **{sel['selective_n']:,}**  "
        f"selective_accuracy = **{sel['selective_accuracy']:.4f}**"
    )
    lines.append("")
    ab = metrics["abstention"]
    lines.append("### Abstention sensitivity / specificity")
    lines.append(
        f"- sensitivity (abstain when supplied says UNANSWERABLE) = **{_fmt(ab.get('sensitivity'))}**\n"
        f"- specificity (answer when supplied says LEFT/RIGHT)     = **{_fmt(ab.get('specificity'))}**"
    )
    lines.append("")
    lines.append("### Confidence-stratified accuracy")
    lines.append("| Bin | n | accuracy | coverage_of_bin |")
    lines.append("|---|---:|---:|---:|")
    for c in metrics["confidence_stratified"]:
        lines.append(
            f"| {c['bin']} | {c.get('n', 0)} | {_fmt(c.get('accuracy'))} | "
            f"{c.get('coverage_of_bin', '—')} |"
        )
    lines.append("")
    ra = metrics["reviewer_agreement"]
    lines.append("### Reviewer agreement")
    lines.append(f"- Cohen's κ = **{_fmt(ra['cohens_kappa'].get('kappa'))}** "
                 f"(n = {ra['cohens_kappa'].get('n', 0)})")
    lines.append(f"- Krippendorff's α = **{_fmt(ra['krippendorffs_alpha'].get('alpha'))}**")
    lines.append("")
    b = metrics["clustered_bootstrap_accuracy_ci"]
    lines.append("### Clustered bootstrap 95 % CI (accuracy on unilateral supplied labels)")
    lines.append(
        f"- point = **{b['point']:.4f}**  95 % CI = **[{b['ci_low']:.4f}, {b['ci_high']:.4f}]**  "
        f"(n_iter = {b['iterations']})"
    )
    lines.append("")
    lines.append("## Claim verification")
    lines.append(_claims_table(metrics["claim_verification"]))
    lines.append("")
    lines.append("## Limitations")
    lines.append(
        "- The blinded model is not a clinical instrument. Every model output should be "
        "treated as a hypothesis for human review, not a diagnosis.\n"
        "- Perceptual-hash near-duplicate grouping cannot detect all image manipulations "
        "(heavy recolouring, non-affine crops, semantic edits).\n"
        "- Supplied labels may themselves be wrong; agreement statistics reflect concordance "
        "with the supplied labels, not with clinical truth.\n"
        "- Reviewer agreement metrics require both reviewers to have covered the same rows.\n"
        "- Clustered bootstrap CIs assume `source_group_id` is the only leakage boundary; "
        "patient-level linkage, if available separately, should be added as an additional cluster."
    )
    lines.append("")
    lines.append("## Final statement")
    lines.append(_final_statement(metrics))
    md.write_text("\n".join(lines))
    return md


def build_paper_tables_and_figures(metrics: dict[str, Any], out_dir: Path) -> list[Path]:
    """Emit paper-ready CSV tables and one PNG figure (label distribution)."""
    tables_dir = ensure_dir(out_dir / "tables")
    figures_dir = ensure_dir(out_dir / "figures")
    outputs: list[Path] = []

    # Table 1: denominators
    t1 = pd.DataFrame([metrics["denominators"]]).T.reset_index()
    t1.columns = ["quantity", "value"]
    t1.to_csv(tables_dir / "table_1_denominators.csv", index=False)
    outputs.append(tables_dir / "table_1_denominators.csv")

    # Table 2: label distribution
    ld = metrics["label_distribution"]
    t2 = pd.DataFrame([
        {"label": k, "count": v["count"], "rate": v["rate"], "denominator": v["denominator"]}
        for k, v in ld.items()
    ])
    t2.to_csv(tables_dir / "table_2_label_distribution.csv", index=False)
    outputs.append(tables_dir / "table_2_label_distribution.csv")

    # Table 3: claim verification
    t3 = pd.DataFrame(metrics["claim_verification"])
    t3.to_csv(tables_dir / "table_3_claim_verification.csv", index=False)
    outputs.append(tables_dir / "table_3_claim_verification.csv")

    # Figure 1: label distribution bar chart (matplotlib-free — use PIL)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(list(ld.keys()), [v["count"] for v in ld.values()])
        ax.set_ylabel("count")
        ax.set_title("Label distribution (final)")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        p = figures_dir / "fig_1_label_distribution.png"
        plt.savefig(p, dpi=150)
        plt.close(fig)
        outputs.append(p)
    except Exception:
        pass
    return outputs


def write_run_manifest(db: Database, cfg: AppConfig, out_dir: Path) -> Path:
    manifest = {
        "run_id": db.get_meta("run_id"),
        "started_at": db.get_meta("started_at"),
        "finished_at": db.get_meta("finished_at") or utc_now_iso(),
        "dermlat_audit_version": __version__,
        "pipeline_version": PIPELINE_VERSION,
        "backend_name": db.get_meta("backend_name"),
        "backend_model": db.get_meta("backend_model"),
        "backend_version": db.get_meta("backend_version"),
        "prompt_hash": db.get_meta("prompt_hash"),
        "hardware": db.get_meta("hardware"),
        "config": cfg.model_dump(mode="json"),
        "preflight": db.get_meta("preflight"),
        "dedup": db.get_meta("dedup"),
        "inference_pass_1": db.get_meta("inference_pass_1"),
        "inference_pass_2": db.get_meta("inference_pass_2"),
    }
    p = out_dir / "run_manifest.json"
    p.write_text(json.dumps(manifest, indent=2, default=str))
    return p


def build_all(cfg: AppConfig, out_dir: Path) -> dict[str, Any]:
    ensure_dir(out_dir)
    db_path = out_dir / "audit.sqlite"
    with Database(db_path) as db:
        csvs = write_csvs(db, out_dir)
        splits_df, splits_ok, splits_note = build_splits(db, cfg, out_dir)
        metrics = compute_metrics(db, cfg)
        metrics["leakage_check"] = {"ok": splits_ok, "note": splits_note,
                                    "n_rows": int(splits_df.shape[0])}
        # Attach optional experiment results if present.
        for key in ("adversarial_comparison", "adversarial_leaky_pass",
                    "injection_scores", "injection_ground_truth_summary"):
            v = db.get_meta(key)
            if v is not None:
                metrics[key] = v
        # Paper-ready figures + calibration.
        paper_extra = render_paper_figures(db, metrics, out_dir)
        metrics.update(paper_extra)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        md = build_report(cfg, metrics, out_dir)
        artefacts = build_paper_tables_and_figures(metrics, out_dir)
        manifest = write_run_manifest(db, cfg, out_dir)
        return {"metrics": metrics, "report_md": str(md),
                "csvs": {k: str(v) for k, v in csvs.items()},
                "artefacts": [str(a) for a in artefacts],
                "run_manifest": str(manifest)}
