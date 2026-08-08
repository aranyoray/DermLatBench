"""dermlat-audit: local, reproducible audit pipeline for DermLatBench."""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("dermlat-audit")
except _metadata.PackageNotFoundError:
    __version__ = "0.0.0+local"

PIPELINE_VERSION = "1.0.0"

__all__ = ["PIPELINE_VERSION", "__version__"]
