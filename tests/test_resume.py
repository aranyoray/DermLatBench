from __future__ import annotations

from pathlib import Path

from dermlat_audit.config import AppConfig, PathsConfig
from dermlat_audit.db import Database
from dermlat_audit.inference.mock_backend import MockBackend
from dermlat_audit.pipeline import run_inference, run_preflight


def test_resume_only_processes_remaining(tmp_path: Path, synthetic_dataset):
    cfg = AppConfig(
        paths=PathsConfig(images_dir=synthetic_dataset["images_dir"],
                          annotations=synthetic_dataset["annotations_csv"],
                          output_dir=tmp_path),
        backend="mock",
    )
    with Database(tmp_path / "audit.sqlite") as db:
        run_preflight(cfg, db)
        backend = MockBackend()
        r1 = run_inference(cfg, db, backend, pass_number=1)
        # A second run should process zero images (all already have pass_1 predictions).
        r2 = run_inference(cfg, db, backend, pass_number=1)
        assert r1["processed"] > 0
        assert r2["processed"] == 0


def test_resume_after_simulated_interruption(tmp_path: Path, synthetic_dataset):
    cfg = AppConfig(
        paths=PathsConfig(images_dir=synthetic_dataset["images_dir"],
                          annotations=synthetic_dataset["annotations_csv"],
                          output_dir=tmp_path),
        backend="mock",
    )
    backend = MockBackend()

    # First "run" — simulate crash by only processing the first few readable images.
    with Database(tmp_path / "audit.sqlite") as db:
        run_preflight(cfg, db)
        readable = [r for r in db.readable_images_without_predictions(backend.model_name)]
        assert len(readable) > 3
        # Insert predictions for the first 3 directly, then abandon.
        for r in readable[:3]:
            result = backend.infer(Path(r["path"]), "p", timeout_seconds=5)
            db.insert_prediction({
                "path": r["path"], "model_name": backend.model_name,
                "model_version": backend.model_version, "pass_number": 1,
                "pipeline_version": "test", "prompt_hash": "hash",
                "raw_response": result.raw_text, "parsed_json": result.raw_text,
                "laterality": "LEFT", "answerable": 1, "confidence": 70,
                "anatomical_site": None, "visible_evidence": "x",
                "ambiguity_reason": None, "review_required": 0,
                "possible_label_error": 0, "flags": "[]",
                "valid": 1, "error": None, "processed_at": "t", "duration_ms": 1,
            })

    # Second "run" — resumes.
    with Database(tmp_path / "audit.sqlite") as db:
        r2 = run_inference(cfg, db, backend, pass_number=1)
        assert r2["processed"] == len(readable) - 3
        # A third run must be idempotent.
        r3 = run_inference(cfg, db, backend, pass_number=1)
        assert r3["processed"] == 0
