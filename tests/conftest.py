"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.synthetic.generate import generate


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict:
    dst = tmp_path_factory.mktemp("synthetic_ds")
    return generate(dst, n_per_class=4)


@pytest.fixture()
def tmp_output(tmp_path: Path) -> Path:
    d = tmp_path / "out"
    d.mkdir(parents=True, exist_ok=True)
    return d
