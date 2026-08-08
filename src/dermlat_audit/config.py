"""Config loader — YAML + env + CLI overrides. Validated via Pydantic."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    images_dir: Path
    annotations: Path | None = None
    manuscript_dir: Path | None = None
    output_dir: Path

    @field_validator("images_dir", "output_dir", "annotations", "manuscript_dir", mode="before")
    @classmethod
    def _expand(cls, v: Any) -> Any:
        if v is None or v == "":
            return None
        return Path(os.path.expandvars(os.path.expanduser(str(v)))).resolve()


class MlxConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    max_new_tokens: int = 512
    temperature: float = 0.0


class OllamaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen2-vl:7b"
    request_timeout_seconds: int = 90


class DedupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    phash_bits: int = 64
    near_duplicate_threshold: int = 6
    bucket_prefix_bits: int = 16


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workers: int | Literal["auto"] = "auto"
    batch_size: int | Literal["auto"] = "auto"
    low_confidence_threshold: int = 60
    max_json_retries: int = 3
    per_image_timeout_seconds: int = 45


class SplitsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15
    random_seed: int = 20260808

    @field_validator("test")
    @classmethod
    def _sum_to_one(cls, v: float, info: Any) -> float:
        train = info.data.get("train", 0.0)
        val = info.data.get("val", 0.0)
        total = train + val + v
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"splits must sum to 1.0 (got {total})")
        return v


class StatsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bootstrap_iterations: int = 2000
    bootstrap_seed: int = 20260808


class ReviewConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    port: int = 8765
    show_only_flagged: bool = True
    blind_reviewer_2: bool = True


class ClaimsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total_images: int | None = 12541
    answerable: int | None = 9770
    unanswerable: int | None = 2771
    min_annotation_agreement: float | None = 0.95


class AppConfig(BaseModel):
    """Top-level application config."""

    model_config = ConfigDict(extra="forbid")
    paths: PathsConfig
    backend: Literal["mlx", "ollama", "mock"] = "mlx"
    mlx: MlxConfig = MlxConfig()
    ollama: OllamaConfig = OllamaConfig()
    dedup: DedupConfig = DedupConfig()
    inference: InferenceConfig = InferenceConfig()
    splits: SplitsConfig = SplitsConfig()
    stats: StatsConfig = StatsConfig()
    review: ReviewConfig = ReviewConfig()
    claims: ClaimsConfig = ClaimsConfig()


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Layer DERMLAT_* env vars on top of loaded YAML."""
    paths = data.setdefault("paths", {})
    env_map = {
        "DERMLAT_IMAGES_DIR": ("paths", "images_dir"),
        "DERMLAT_ANNOTATIONS_CSV": ("paths", "annotations"),
        "DERMLAT_MANUSCRIPT_DIR": ("paths", "manuscript_dir"),
        "DERMLAT_OUTPUT_DIR": ("paths", "output_dir"),
        "DERMLAT_BACKEND": ("backend",),
        "DERMLAT_MLX_MODEL": ("mlx", "model"),
        "DERMLAT_OLLAMA_HOST": ("ollama", "host"),
        "DERMLAT_OLLAMA_MODEL": ("ollama", "model"),
    }
    for env, path in env_map.items():
        v = os.environ.get(env)
        if not v:
            continue
        cursor: Any = data
        for key in path[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[path[-1]] = v
    # Silence unused-var warnings on py311 dict linters.
    del paths
    return data


def load_config(
    config_path: Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Load config from YAML file (optional), apply env vars, apply flat overrides.

    ``overrides`` may use dotted keys, e.g. ``{"paths.images_dir": "/x"}``.
    """
    data: dict[str, Any] = {}
    if config_path and config_path.exists():
        with config_path.open() as fh:
            loaded = yaml.safe_load(fh) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"config at {config_path} is not a mapping")
            data = loaded

    data = _apply_env_overrides(data)

    if overrides:
        for dotted, value in overrides.items():
            if value is None:
                continue
            cursor: Any = data
            keys = dotted.split(".")
            for k in keys[:-1]:
                cursor = cursor.setdefault(k, {})
            cursor[keys[-1]] = value

    return AppConfig(**data)
