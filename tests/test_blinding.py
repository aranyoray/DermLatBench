"""Verify that supplied labels, filenames, and folders cannot leak into blinded inference."""

from __future__ import annotations

import inspect
from pathlib import Path

from PIL import Image

from dermlat_audit.inference.mlx_backend import MLXBackend
from dermlat_audit.inference.mock_backend import MockBackend
from dermlat_audit.inference.ollama_backend import OllamaBackend
from dermlat_audit.prompts import BLINDED_LATERALITY_PROMPT, prompt_text


def _synthetic_image(tmp_path: Path) -> Path:
    p = tmp_path / "LEFT_hint_in_filename_reviewer_1.png"
    Image.new("RGB", (32, 32), color=(140, 90, 90)).save(p)
    return p


def test_prompt_contains_no_leak_fields():
    """The prompt itself must not mention filename, folder, EXIF, or supplied labels."""
    p = prompt_text().lower()
    for banned in ("filename", "file name", "folder", "directory", "exif",
                   "supplied label", "existing annotation"):
        assert banned in p or f"no {banned}" in p or "no filename" in p, banned
    # There's an explicit rule saying "You have no filename..."
    assert "no filename" in p


def test_mock_backend_ignores_prompt_content(tmp_path: Path):
    """The mock backend's output should depend only on pixel content — not on any prompt string."""
    img = _synthetic_image(tmp_path)
    b = MockBackend()
    a = b.infer(img, "prompt A", timeout_seconds=5)
    z = b.infer(img, "totally different prompt Z with LEFT and RIGHT hints", timeout_seconds=5)
    assert a.raw_text == z.raw_text


def test_backend_signatures_take_only_image_and_prompt():
    """No backend should ask for supplied labels or filename metadata."""
    for cls in (MockBackend, OllamaBackend, MLXBackend):
        sig = inspect.signature(cls.infer)
        assert set(sig.parameters) == {"self", "image_path", "prompt", "timeout_seconds"}


def test_pipeline_never_passes_supplied_label_to_backend(tmp_path: Path, synthetic_dataset):
    """The pipeline's inference loop must call backend.infer with only path+prompt."""
    from dermlat_audit.config import AppConfig, PathsConfig
    from dermlat_audit.db import Database
    from dermlat_audit.pipeline import run_inference, run_preflight

    cfg = AppConfig(paths=PathsConfig(
        images_dir=synthetic_dataset["images_dir"],
        annotations=synthetic_dataset["annotations_csv"],
        output_dir=tmp_path,
    ), backend="mock")

    seen_prompts: list[str] = []

    class SpyBackend(MockBackend):
        def infer(self, image_path, prompt, *, timeout_seconds):
            seen_prompts.append(prompt)
            return super().infer(image_path, prompt, timeout_seconds=timeout_seconds)

    with Database(tmp_path / "audit.sqlite") as db:
        run_preflight(cfg, db)
        run_inference(cfg, db, SpyBackend(), pass_number=1)

    assert seen_prompts, "expected at least one inference call"
    for p in seen_prompts:
        assert p == BLINDED_LATERALITY_PROMPT
        # None of the supplied labels or filenames should appear.
        for banned in ("LEFT_hint_in_filename", "annotations.csv", "supplied_label"):
            assert banned not in p
