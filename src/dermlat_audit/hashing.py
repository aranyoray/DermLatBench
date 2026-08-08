"""SHA-256 and perceptual-hash helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, UnidentifiedImageError

_READ_CHUNK = 1024 * 1024  # 1 MB


def sha256_file(path: Path) -> str:
    """Streaming SHA-256 of a file's bytes on disk."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_READ_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def phash_hex(path: Path, hash_size: int = 8) -> str | None:
    """Perceptual-hash of an image, returned as a lowercase hex string.

    Rotation- and mirror-invariance is handled by the deduper, not here.
    Returns None on unreadable images (caller records the failure).
    """
    try:
        import imagehash  # local import — imagehash pulls scipy on some platforms
    except ImportError as exc:  # pragma: no cover - hard install error
        raise RuntimeError("imagehash is required") from exc
    try:
        with Image.open(path) as im:
            im.load()
            return str(imagehash.phash(im.convert("RGB"), hash_size=hash_size))
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def hamming_hex(a: str, b: str) -> int:
    """Bit-count of the XOR between two equal-length hex hash strings."""
    if len(a) != len(b):
        raise ValueError(f"hash length mismatch: {len(a)} vs {len(b)}")
    return bin(int(a, 16) ^ int(b, 16)).count("1")
