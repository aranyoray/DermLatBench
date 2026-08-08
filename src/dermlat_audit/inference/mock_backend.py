"""Deterministic offline backend for tests, CI, and smoke runs.

Its output depends ONLY on image pixel content (mean intensity of the left vs right half),
so tests can exercise the pipeline without any AI dependency and still get repeatable,
schema-valid output.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .base import Backend, InferenceError, InferenceResult


class MockBackend(Backend):
    name = "mock"

    @property
    def model_name(self) -> str:
        return "mock-image-side-vlm"

    @property
    def model_version(self) -> str:
        return "1.0.0"

    def infer(self, image_path: Path, prompt: str, *, timeout_seconds: int) -> InferenceResult:
        # timeout_seconds accepted for interface parity; we're fast enough to ignore it.
        del prompt, timeout_seconds
        t0 = time.perf_counter()
        try:
            with Image.open(image_path) as im:
                arr = np.asarray(im.convert("L"))
        except Exception as exc:
            raise InferenceError(f"mock backend failed to read image: {exc}") from exc
        if arr.size == 0:
            raise InferenceError("empty image")
        _h, w = arr.shape
        left_mean = float(arr[:, : w // 2].mean())
        right_mean = float(arr[:, w // 2 :].mean())
        diff = right_mean - left_mean
        if abs(diff) < 3.0:
            answer = _pack("BILATERAL_OR_MIDLINE", answerable=True,
                           evidence="Left and right halves have similar intensity, suggesting a midline or bilateral pattern; visible orienting midline landmark assumed.",
                           confidence=55)
        elif abs(diff) < 8.0:
            answer = _pack("UNANSWERABLE", answerable=False,
                           evidence="Uncertain image composition; no reliable orienting anatomy visible.",
                           confidence=30, review_required=True)
        else:
            side = "LEFT" if diff > 0 else "RIGHT"
            answer = _pack(side, answerable=True,
                           evidence=f"Contralateral limb intensity asymmetry suggests patient's {side} side is visible.",
                           confidence=75)
        dt_ms = int((time.perf_counter() - t0) * 1000)
        return InferenceResult(raw_text=json.dumps(answer), model_name=self.model_name,
                               model_version=self.model_version, duration_ms=dt_ms)


def _pack(
    laterality: str,
    *,
    answerable: bool,
    evidence: str,
    confidence: int,
    review_required: bool = False,
    possible_label_error: bool = False,
    anatomical_site: str | None = "unspecified",
    ambiguity_reason: str | None = None,
) -> dict:
    return {
        "laterality": laterality,
        "anatomical_site": anatomical_site,
        "answerable": answerable,
        "visible_evidence": evidence,
        "confidence": confidence,
        "ambiguity_reason": ambiguity_reason,
        "review_required": review_required,
        "possible_label_error": possible_label_error,
    }
