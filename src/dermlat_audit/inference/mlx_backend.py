"""Apple-Silicon-native MLX-VLM backend.

Lazily imported so machines without mlx-vlm can still run the mock/ollama backends and
the test suite. The image is opened here and passed to the model as pixels — filename,
folder, and EXIF never enter the prompt.
"""

from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from .base import Backend, InferenceError, InferenceResult


class MLXBackend(Backend):
    name = "mlx"

    def __init__(self, model_id: str, max_new_tokens: int = 512, temperature: float = 0.0):
        self._model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._processor = None
        self._version: str | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from mlx_vlm import load
            from mlx_vlm.version import __version__ as mlx_vlm_version
        except ImportError as exc:  # pragma: no cover - hardware dep
            raise InferenceError(
                "mlx-vlm is not installed. Install with:  pip install 'dermlat-audit[mlx]'"
            ) from exc
        self._model, self._processor = load(self._model_id)
        self._version = mlx_vlm_version

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def model_version(self) -> str:
        return f"mlx-vlm={self._version}" if self._version else "mlx-vlm=unknown"

    def infer(self, image_path: Path, prompt: str, *, timeout_seconds: int) -> InferenceResult:
        # timeout_seconds is best-effort; MLX doesn't offer a graceful in-process abort,
        # so this is enforced by the pipeline's per-image supervisor when needed.
        del timeout_seconds
        self._ensure_loaded()
        try:
            from mlx_vlm import generate
            from mlx_vlm.prompt_utils import apply_chat_template
        except ImportError as exc:  # pragma: no cover
            raise InferenceError("mlx-vlm generate not available") from exc

        # Verify we can open the image before handing it to the model.
        try:
            with Image.open(image_path) as _im:
                _im.load()
        except Exception as exc:
            raise InferenceError(f"cannot open image: {exc}") from exc

        formatted = apply_chat_template(
            self._processor, config=getattr(self._model, "config", None),
            prompt=prompt, num_images=1,
        )

        t0 = time.perf_counter()
        try:
            output = generate(
                self._model,
                self._processor,
                formatted,
                [str(image_path)],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                verbose=False,
            )
        except Exception as exc:
            raise InferenceError(f"mlx-vlm generate failed: {exc}") from exc
        dt_ms = int((time.perf_counter() - t0) * 1000)

        text = output if isinstance(output, str) else getattr(output, "text", str(output))
        return InferenceResult(raw_text=text, model_name=self.model_name,
                               model_version=self.model_version, duration_ms=dt_ms)
