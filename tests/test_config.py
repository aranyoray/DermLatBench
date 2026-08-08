from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dermlat_audit.config import AppConfig, PathsConfig, SplitsConfig, load_config


def test_loads_from_yaml(tmp_path: Path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        f"paths:\n  images_dir: {tmp_path}\n  output_dir: {tmp_path / 'out'}\nbackend: mock\n"
    )
    cfg = load_config(cfg_path)
    assert cfg.backend == "mock"
    assert cfg.paths.images_dir == tmp_path


def test_env_override(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        f"paths:\n  images_dir: {tmp_path}\n  output_dir: {tmp_path}\nbackend: mock\n"
    )
    monkeypatch.setenv("DERMLAT_BACKEND", "mlx")
    cfg = load_config(cfg_path)
    assert cfg.backend == "mlx"


def test_splits_must_sum_to_one():
    with pytest.raises(ValidationError):
        SplitsConfig(train=0.5, val=0.3, test=0.3)


def test_extra_key_rejected(tmp_path: Path):
    with pytest.raises(ValidationError):
        AppConfig(paths=PathsConfig(images_dir=tmp_path, output_dir=tmp_path),
                  unknown_key=True)  # type: ignore[call-arg]
