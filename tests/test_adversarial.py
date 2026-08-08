from __future__ import annotations

from pathlib import Path

from dermlat_audit.adversarial import (
    BLINDED_LATERALITY_PROMPT,
    compare_blinded_vs_leaky,
    leaky_prompt,
    leaky_prompt_hash,
    run_leaky_pass,
)
from dermlat_audit.config import AppConfig, PathsConfig
from dermlat_audit.db import Database
from dermlat_audit.inference.mock_backend import MockBackend
from dermlat_audit.pipeline import run_full
from dermlat_audit.prompts import prompt_hash


def test_leaky_prompt_extends_blinded():
    p = leaky_prompt("foo.png", "left")
    assert p.startswith(BLINDED_LATERALITY_PROMPT)
    assert "foo.png" in p
    assert "left" in p


def test_leaky_prompt_hash_differs_from_blinded():
    assert leaky_prompt_hash() != prompt_hash()


def test_run_leaky_pass_and_compare(tmp_path: Path, synthetic_dataset):
    cfg = AppConfig(paths=PathsConfig(images_dir=synthetic_dataset["images_dir"],
                                       annotations=synthetic_dataset["annotations_csv"],
                                       output_dir=tmp_path),
                    backend="mock")
    run_full(cfg, tmp_path)
    with Database(tmp_path / "audit.sqlite") as db:
        result = run_leaky_pass(cfg, db, MockBackend())
        assert result["processed"] > 0
        cmp = compare_blinded_vs_leaky(db)
        assert cmp["blinded"]["n"] > 0
        assert cmp["leaky"]["n"] > 0
        # Same backend, same pixels — accuracy should be identical for mock.
        assert cmp["blinded"]["accuracy"] == cmp["leaky"]["accuracy"]


def test_prompt_still_uses_the_frozen_blinded_prompt():
    """The blinded run in the pipeline must never use the leaky variant."""
    from dermlat_audit.prompts import prompt_text
    assert prompt_text() == BLINDED_LATERALITY_PROMPT
    assert "filename" not in prompt_text() or "no filename" in prompt_text()
