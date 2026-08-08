from __future__ import annotations

from dermlat_audit.config import ClaimsConfig
from dermlat_audit.verify_claims import overall_publication_ready, verify_claims


def test_verify_verified_when_matches():
    r = verify_claims(
        ClaimsConfig(total_images=100, answerable=70, unanswerable=30,
                     min_annotation_agreement=0.95),
        observed_total=100, observed_answerable=70, observed_unanswerable=30,
        observed_agreement=0.97,
    )
    statuses = {c.claim: c.status for c in r}
    assert statuses["total_images"] == "VERIFIED"
    assert statuses["answerable"] == "VERIFIED"
    assert statuses["unanswerable"] == "VERIFIED"
    assert statuses["annotation_agreement>threshold"] == "VERIFIED"
    assert overall_publication_ready(r)


def test_verify_contradicted_on_mismatch():
    r = verify_claims(
        ClaimsConfig(total_images=100, answerable=70, unanswerable=30,
                     min_annotation_agreement=0.95),
        observed_total=200, observed_answerable=60, observed_unanswerable=30,
        observed_agreement=0.80,
    )
    statuses = {c.claim: c.status for c in r}
    assert statuses["total_images"] == "CONTRADICTED"
    assert statuses["answerable"] == "CONTRADICTED"
    assert statuses["annotation_agreement>threshold"] == "CONTRADICTED"
    assert not overall_publication_ready(r)


def test_verify_not_verifiable_when_no_observation():
    r = verify_claims(
        ClaimsConfig(total_images=100, answerable=70, unanswerable=30,
                     min_annotation_agreement=0.95),
        observed_total=100, observed_answerable=70, observed_unanswerable=30,
        observed_agreement=None,
    )
    statuses = {c.claim: c.status for c in r}
    assert statuses["annotation_agreement>threshold"] == "NOT_VERIFIABLE"
    assert not overall_publication_ready(r)
