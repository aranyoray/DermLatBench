"""Preflight & inventory: discover every image, record its physical properties."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import ExifTags, Image, UnidentifiedImageError

from .constants import SUPPORTED_EXTENSIONS
from .hashing import sha256_file
from .utils import utc_now_iso


@dataclass(frozen=True, slots=True)
class ManifestRow:
    path: str                    # absolute path
    relpath: str                 # path relative to images_dir
    sha256: str | None
    size_bytes: int
    width: int | None
    height: int | None
    format: str | None
    mode: str | None
    exif_present: bool
    supported: bool
    readable: bool
    error: str | None
    discovered_at: str


def _read_image_meta(path: Path) -> tuple[bool, tuple[int | None, int | None, str | None, str | None, bool], str | None]:
    """Return (readable, (w,h,fmt,mode,exif), error_or_none)."""
    try:
        with Image.open(path) as im:
            im.load()
            w, h = im.size
            fmt = (im.format or "").upper() or None
            mode = im.mode
            exif_present = False
            try:
                exif = im.getexif()
                exif_present = bool(exif) and any(exif.values())
            except Exception:
                exif_present = False
            # silence unused
            _ = ExifTags
            return True, (w, h, fmt, mode, exif_present), None
    except UnidentifiedImageError as exc:
        return False, (None, None, None, None, False), f"unreadable: {exc}"
    except OSError as exc:
        return False, (None, None, None, None, False), f"os_error: {exc}"


def discover_images(images_dir: Path, hash_files: bool = True) -> list[ManifestRow]:
    """Recursively walk ``images_dir`` and return one row per file found.

    No image is silently skipped. Unsupported extensions and unreadable files are still
    recorded, with ``supported`` / ``readable`` flagging why.
    """
    images_dir = images_dir.resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images_dir does not exist: {images_dir}")

    rows: list[ManifestRow] = []
    for path in sorted(images_dir.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        supported = ext in SUPPORTED_EXTENSIONS
        size_bytes = path.stat().st_size
        readable, (w, h, fmt, mode, exif_present), err = (
            _read_image_meta(path) if supported else (False, (None, None, None, None, False), "unsupported_extension")
        )
        sha = sha256_file(path) if (hash_files and readable) else None
        rows.append(
            ManifestRow(
                path=str(path),
                relpath=str(path.relative_to(images_dir)),
                sha256=sha,
                size_bytes=size_bytes,
                width=w,
                height=h,
                format=fmt,
                mode=mode,
                exif_present=exif_present,
                supported=supported,
                readable=readable,
                error=err,
                discovered_at=utc_now_iso(),
            )
        )
    return rows


def summarize_manifest(rows: list[ManifestRow]) -> dict[str, int]:
    """One-line inventory summary."""
    total = len(rows)
    supported = sum(1 for r in rows if r.supported)
    readable = sum(1 for r in rows if r.readable)
    corrupt = sum(1 for r in rows if r.supported and not r.readable)
    unsupported = total - supported
    return {
        "total_discovered": total,
        "supported": supported,
        "readable": readable,
        "corrupt": corrupt,
        "unsupported": unsupported,
    }
