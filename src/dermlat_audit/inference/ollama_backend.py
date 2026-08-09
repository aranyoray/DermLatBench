"""Ollama backend — optional fallback."""

from __future__ import annotations

import base64
import time
from pathlib import Path

from .base import Backend, InferenceError, InferenceResult


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, host: str, model: str, request_timeout_seconds: int = 90):
        self.host = host.rstrip("/")
        self._model = model
        self.request_timeout_seconds = request_timeout_seconds
        self._version: str | None = None
        self._client = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise InferenceError(
                "httpx is not installed. Install with:  pip install 'dermlat-audit[ollama]'"
            ) from exc
        self._client = httpx.Client(base_url=self.host, timeout=self.request_timeout_seconds)
        try:
            r = self._client.get("/api/version")
            r.raise_for_status()
            self._version = r.json().get("version", "unknown")
        except Exception as exc:
            raise InferenceError(f"cannot reach ollama at {self.host}: {exc}") from exc

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def model_version(self) -> str:
        return f"ollama={self._version}" if self._version else "ollama=unknown"

    def infer(self, image_path: Path, prompt: str, *, timeout_seconds: int) -> InferenceResult:
        self._ensure_client()
        assert self._client is not None
        try:
            b = image_path.read_bytes()
        except OSError as exc:
            raise InferenceError(f"cannot open image: {exc}") from exc

        payload = {
            "model": self._model,
            "prompt": prompt,
            "images": [base64.b64encode(b).decode("ascii")],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 384},
        }
        t0 = time.perf_counter()
        try:
            r = self._client.post("/api/generate", json=payload, timeout=timeout_seconds)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            raise InferenceError(f"ollama request failed: {exc}") from exc
        dt_ms = int((time.perf_counter() - t0) * 1000)
        text = data.get("response") or ""
        return InferenceResult(raw_text=text, model_name=self.model_name,
                               model_version=self.model_version, duration_ms=dt_ms)
