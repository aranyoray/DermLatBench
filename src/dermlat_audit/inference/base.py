"""Pluggable local backend interface + factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import AppConfig


class InferenceError(RuntimeError):
    """Raised on backend failure (transport, timeout, model crash)."""


@dataclass(slots=True)
class InferenceResult:
    raw_text: str            # exactly what the model returned
    model_name: str
    model_version: str
    duration_ms: int


class Backend(ABC):
    """A blinded local VLM. Sees only image pixels + the prompt string."""

    name: str = "abstract"

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def model_version(self) -> str: ...

    @abstractmethod
    def infer(self, image_path: Path, prompt: str, *, timeout_seconds: int) -> InferenceResult:
        """Run the model on one image. Must not read filename/EXIF into the prompt."""


def make_backend(cfg: AppConfig) -> Backend:
    """Instantiate the backend named in ``cfg.backend``."""
    match cfg.backend:
        case "mlx":
            from .mlx_backend import MLXBackend
            return MLXBackend(cfg.mlx.model, cfg.mlx.max_new_tokens, cfg.mlx.temperature)
        case "ollama":
            from .ollama_backend import OllamaBackend
            return OllamaBackend(cfg.ollama.host, cfg.ollama.model, cfg.ollama.request_timeout_seconds)
        case "mock":
            from .mock_backend import MockBackend
            return MockBackend()
    raise ValueError(f"unknown backend: {cfg.backend}")
