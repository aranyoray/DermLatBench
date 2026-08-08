from __future__ import annotations

from pathlib import Path

from PIL import Image

from dermlat_audit.config import AppConfig, PathsConfig
from dermlat_audit.db import Database
from dermlat_audit.inference.base import Backend, InferenceResult
from dermlat_audit.pipeline import run_inference, run_preflight


class BadBackend(Backend):
    name = "bad"

    def __init__(self):
        self._i = 0

    @property
    def model_name(self):
        return "bad-model"

    @property
    def model_version(self):
        return "0"

    def infer(self, image_path: Path, prompt: str, *, timeout_seconds: int) -> InferenceResult:
        self._i += 1
        # First: unparseable text. Second: valid JSON but wrong enum. Third: wrong types.
        payloads = [
            "hello, world",
            '{"laterality": "SORTA_LEFT", "answerable": true, "visible_evidence": "x",'
            ' "confidence": 50, "review_required": false, "possible_label_error": false}',
            '{"laterality": "LEFT", "answerable": "yes", "visible_evidence": "x",'
            ' "confidence": 50, "review_required": false, "possible_label_error": false}',
        ]
        text = payloads[(self._i - 1) % len(payloads)]
        return InferenceResult(raw_text=text, model_name=self.model_name,
                               model_version=self.model_version, duration_ms=1)


def _mini_dataset(tmp_path: Path) -> tuple[Path, Path]:
    d = tmp_path / "images"
    d.mkdir()
    for i in range(3):
        Image.new("RGB", (32, 32), color=(120, 40 * i, 40)).save(d / f"im_{i}.png")
    return d, tmp_path / "out"


def test_malformed_responses_flagged_not_skipped(tmp_path: Path):
    imgs, out = _mini_dataset(tmp_path)
    out.mkdir()
    cfg = AppConfig(paths=PathsConfig(images_dir=imgs, output_dir=out), backend="mock")
    with Database(out / "audit.sqlite") as db:
        run_preflight(cfg, db)
        r = run_inference(cfg, db, BadBackend(), pass_number=1)
        # Nothing silently skipped — every image has a prediction row.
        assert r["processed"] == 3
        assert r["flagged"] == 3
        preds = db._conn.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
        assert preds == 3
        # And every one made it into the review queue.
        queue = db.review_queue()
        assert len(queue) == 3
