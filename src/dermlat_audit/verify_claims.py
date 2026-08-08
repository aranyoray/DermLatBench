"""Independent verification of manuscript claims.

Each claim is labelled one of:
  VERIFIED         — our number matches the claim exactly (or within tolerance).
  CONTRADICTED     — our number is definitively outside the claim.
  NOT_VERIFIABLE   — required evidence missing (e.g. no supplied labels available).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import ClaimsConfig


@dataclass
class ClaimResult:
    claim: str
    claim_value: Any
    observed_value: Any
    status: str
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return {"claim": self.claim, "claim_value": self.claim_value,
                "observed_value": self.observed_value,
                "status": self.status, "notes": self.notes}


def _count_status(observed: int | None, expected: int | None, *, tol: int = 0) -> tuple[str, str]:
    if observed is None or expected is None:
        return "NOT_VERIFIABLE", "no observed value"
    if abs(observed - expected) <= tol:
        return "VERIFIED", f"observed={observed} within tolerance ({tol}) of claim={expected}"
    return "CONTRADICTED", f"observed={observed} vs claimed={expected} (diff={observed-expected})"


def _rate_status(observed: float | None, threshold: float | None, *, side: str = "greater") -> tuple[str, str]:
    if observed is None or threshold is None:
        return "NOT_VERIFIABLE", "no observed value"
    if side == "greater":
        if observed > threshold:
            return "VERIFIED", f"observed={observed:.4f} > claimed threshold {threshold:.4f}"
        return "CONTRADICTED", f"observed={observed:.4f} ≤ claimed threshold {threshold:.4f}"
    if observed < threshold:
        return "VERIFIED", f"observed={observed:.4f} < claimed threshold {threshold:.4f}"
    return "CONTRADICTED", f"observed={observed:.4f} ≥ claimed threshold {threshold:.4f}"


def verify_claims(
    claims: ClaimsConfig,
    observed_total: int,
    observed_answerable: int,
    observed_unanswerable: int,
    observed_agreement: float | None,
) -> list[ClaimResult]:
    results: list[ClaimResult] = []
    status, notes = _count_status(observed_total, claims.total_images)
    results.append(ClaimResult("total_images", claims.total_images, observed_total, status, notes))

    status, notes = _count_status(observed_answerable, claims.answerable)
    results.append(ClaimResult("answerable", claims.answerable, observed_answerable, status, notes))

    status, notes = _count_status(observed_unanswerable, claims.unanswerable)
    results.append(ClaimResult("unanswerable", claims.unanswerable, observed_unanswerable, status, notes))

    status, notes = _rate_status(observed_agreement, claims.min_annotation_agreement, side="greater")
    results.append(ClaimResult(
        "annotation_agreement>threshold", claims.min_annotation_agreement,
        observed_agreement, status, notes,
    ))
    return results


def overall_publication_ready(results: list[ClaimResult]) -> bool:
    """True iff every claim is VERIFIED (or the claim was None so not requested)."""
    for r in results:
        if r.claim_value is None:
            continue
        if r.status != "VERIFIED":
            return False
    return True
