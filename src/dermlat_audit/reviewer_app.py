"""Streamlit two-reviewer + adjudication UI.

Launch:
    dermlat-audit review --output /path/to/private-results

Rules enforced:
- Only rows in the flagged queue are shown (unless the user toggles "show all").
- The supplied label is NEVER shown before the reviewer's decision is saved.
- Reviewer 2 does not see Reviewer 1's decision (until adjudication mode).
- Keyboard shortcuts: 1=LEFT, 2=RIGHT, 3=BILATERAL_OR_MIDLINE, 4=UNANSWERABLE,
                      5=EXCLUDE_CORRUPT_OR_NONCLINICAL, N=next, P=previous.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from .constants import ALL_LABELS
from .db import Database
from .utils import utc_now_iso


def _get_db() -> Database:
    output = os.environ.get("DERMLAT_OUTPUT_DIR")
    if not output:
        st.error("Set DERMLAT_OUTPUT_DIR env var to the audit output directory.")
        st.stop()
    db_path = Path(output) / "audit.sqlite"
    if not db_path.exists():
        st.error(f"No audit.sqlite at {db_path}. Run `dermlat-audit run` first.")
        st.stop()
    return Database(db_path)


def _keyboard_js() -> str:
    return """
    <script>
    document.addEventListener('keydown', function(e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        const map = {
            '1': 'btn-LEFT', '2': 'btn-RIGHT', '3': 'btn-BOTH',
            '4': 'btn-UNANS', '5': 'btn-EXCLUDE',
            'n': 'btn-next', 'p': 'btn-prev', 'N': 'btn-next', 'P': 'btn-prev',
        };
        const key = e.key;
        if (key in map) {
            const el = window.parent.document.querySelector('[data-key="' + map[key] + '"] button');
            if (el) el.click();
        }
    });
    </script>
    """


def _label_buttons(key_prefix: str) -> str | None:
    """Render the five buttons; return the label if any was clicked, else None."""
    st.markdown(_keyboard_js(), unsafe_allow_html=True)
    cols = st.columns(5)
    labels = list(ALL_LABELS)
    short = ["LEFT (1)", "RIGHT (2)", "BOTH/MIDLINE (3)", "UNANS (4)", "EXCLUDE (5)"]
    dkeys = ["LEFT", "RIGHT", "BOTH", "UNANS", "EXCLUDE"]
    for i, (label, s, dkey) in enumerate(zip(labels, short, dkeys, strict=True)):
        with cols[i]:
            st.markdown(f'<div data-key="btn-{dkey}">', unsafe_allow_html=True)
            if st.button(s, key=f"{key_prefix}_{label}", use_container_width=True):
                st.markdown("</div>", unsafe_allow_html=True)
                return label
            st.markdown("</div>", unsafe_allow_html=True)
    return None


def _queue(db: Database, show_all: bool) -> list[dict]:
    if show_all:
        rows = [dict(r) for r in db.all_images() if r["readable"]]
    else:
        rows = [dict(r) for r in db.review_queue()]
    return rows


def _already_reviewed_paths(db: Database, reviewer_id: str) -> set[str]:
    return {r["path"] for r in db.reviews_by_reviewer(reviewer_id)}


def _blind_display_path(path: str) -> str:
    """Show only that a file exists — never leak the filename."""
    return f"image #{abs(hash(path)) % 10_000_000:07d}"


def _reviewer_view(db: Database, reviewer_id: str, *, blind_r1: bool) -> None:
    st.subheader(f"Reviewer: {reviewer_id}")
    show_all = st.checkbox("Show all readable images (not just flagged)", value=False)
    show_supplied = st.checkbox("Show supplied label after I save (audit reference only)", value=True)

    queue = _queue(db, show_all)
    already = _already_reviewed_paths(db, reviewer_id)
    remaining = [r for r in queue if r["path"] not in already]

    st.write(f"Reviewed so far: **{len(already)}**  |  In queue: **{len(remaining)}**")

    if not remaining:
        st.success("Nothing left to review.")
        return

    idx = st.session_state.setdefault(f"{reviewer_id}_idx", 0)
    idx = max(0, min(idx, len(remaining) - 1))
    row = remaining[idx]

    st.write(f"Current: {_blind_display_path(row['path'])}  ({idx + 1}/{len(remaining)})")
    try:
        st.image(row["path"], use_container_width=True)
    except Exception as exc:
        st.error(f"could not display image: {exc}")

    notes = st.text_input("Notes (optional)", key=f"{reviewer_id}_notes_{row['path']}")
    site = st.text_input("Anatomical site", key=f"{reviewer_id}_site_{row['path']}")
    conf = st.slider("Confidence 0–100", 0, 100, 80, key=f"{reviewer_id}_conf_{row['path']}")

    chosen = _label_buttons(f"{reviewer_id}_lbl_{idx}")
    if chosen:
        db.upsert_review({
            "path": row["path"],
            "reviewer_id": reviewer_id,
            "laterality": chosen,
            "answerable": int(chosen in ("LEFT", "RIGHT", "BILATERAL_OR_MIDLINE")),
            "anatomical_site": site or None,
            "visible_evidence": notes or None,
            "confidence": int(conf),
            "notes": notes or None,
            "reviewed_at": utc_now_iso(),
        })
        st.success(f"Saved: {chosen}")
        if show_supplied and row.get("supplied_label"):
            st.info(f"For audit reference (revealed after save): supplied_label = "
                    f"`{row['supplied_label']}`")
        st.session_state[f"{reviewer_id}_idx"] = idx + 1
        st.rerun()

    if blind_r1 and reviewer_id == "reviewer_2":
        st.caption("Reviewer 2 mode: Reviewer 1's decision on this image is hidden.")

    nav = st.columns(2)
    with nav[0]:
        st.markdown('<div data-key="btn-prev">', unsafe_allow_html=True)
        if st.button("← Previous (P)", key=f"{reviewer_id}_prev"):
            st.session_state[f"{reviewer_id}_idx"] = max(0, idx - 1)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with nav[1]:
        st.markdown('<div data-key="btn-next">', unsafe_allow_html=True)
        if st.button("Skip / Next (N) →", key=f"{reviewer_id}_next"):
            st.session_state[f"{reviewer_id}_idx"] = idx + 1
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def _adjudicator_view(db: Database) -> None:
    st.subheader("Adjudicator")
    r1 = {r["path"]: dict(r) for r in db.reviews_by_reviewer("reviewer_1")}
    r2 = {r["path"]: dict(r) for r in db.reviews_by_reviewer("reviewer_2")}
    disagreed = sorted(p for p in (set(r1) & set(r2)) if r1[p]["laterality"] != r2[p]["laterality"])
    only_r1 = sorted(set(r1) - set(r2))
    only_r2 = sorted(set(r2) - set(r1))
    queue = disagreed + only_r1 + only_r2
    st.write(f"Disagreements: **{len(disagreed)}**  Only Reviewer 1: **{len(only_r1)}**  "
             f"Only Reviewer 2: **{len(only_r2)}**")
    if not queue:
        st.success("Nothing to adjudicate.")
        return
    already = {a["path"] for a in db.all_adjudications()}
    remaining = [p for p in queue if p not in already]
    if not remaining:
        st.success("All rows adjudicated.")
        return
    idx = st.session_state.setdefault("adj_idx", 0)
    idx = max(0, min(idx, len(remaining) - 1))
    path = remaining[idx]
    st.write(f"Current: {_blind_display_path(path)}  ({idx + 1}/{len(remaining)})")
    try:
        st.image(path, use_container_width=True)
    except Exception as exc:
        st.error(f"cannot display image: {exc}")

    st.markdown("**Reviewer 1:** " + str(r1.get(path, {}).get("laterality", "—")))
    st.markdown("**Reviewer 2:** " + str(r2.get(path, {}).get("laterality", "—")))
    notes = st.text_input("Adjudication notes", key=f"adj_notes_{path}")
    site = st.text_input("Anatomical site", key=f"adj_site_{path}")
    chosen = _label_buttons(f"adj_lbl_{idx}")
    if chosen:
        db.upsert_adjudication({
            "path": path,
            "laterality": chosen,
            "answerable": int(chosen in ("LEFT", "RIGHT", "BILATERAL_OR_MIDLINE")),
            "anatomical_site": site or None,
            "visible_evidence": notes or None,
            "adjudicator_id": os.environ.get("USER", "adjudicator"),
            "notes": notes or None,
            "adjudicated_at": utc_now_iso(),
        })
        st.success(f"Adjudicated: {chosen}")
        st.session_state["adj_idx"] = idx + 1
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="dermlat-audit reviewer", layout="wide")
    st.title("dermlat-audit reviewer")
    db = _get_db()
    tab = st.sidebar.radio("Mode", ("Reviewer 1", "Reviewer 2", "Adjudicator", "Stats"))
    blind_r1 = os.environ.get("DERMLAT_BLIND_R2", "1") != "0"
    if tab == "Reviewer 1":
        _reviewer_view(db, "reviewer_1", blind_r1=False)
    elif tab == "Reviewer 2":
        _reviewer_view(db, "reviewer_2", blind_r1=blind_r1)
    elif tab == "Adjudicator":
        _adjudicator_view(db)
    else:
        st.write("Run `dermlat-audit report` to (re)build metrics.json + audit_report.md.")
        meta = db.all_meta()
        st.json({k: v for k, v in meta.items() if not isinstance(v, dict) or len(str(v)) < 5000})


if __name__ == "__main__":
    main()
