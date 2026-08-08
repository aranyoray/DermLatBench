"""Local vision-language model backends."""

from .base import Backend, InferenceError, InferenceResult, make_backend

__all__ = ["Backend", "InferenceError", "InferenceResult", "make_backend"]
