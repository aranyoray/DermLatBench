"""Generate synthetic no-PHI test images + a matching annotations CSV.

Deterministic (seeded). Used by both the pytest suite and the smoke run.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw

CANVAS = 64  # small on purpose — tests must be fast


def _base_image(seed: int) -> Image.Image:
    rng = random.Random(seed)
    im = Image.new("RGB", (CANVAS, CANVAS), color=(200, 180, 170))
    d = ImageDraw.Draw(im)
    for _ in range(6):
        x0 = rng.randint(0, CANVAS - 8)
        y0 = rng.randint(0, CANVAS - 8)
        r = rng.randint(2, 6)
        color = (rng.randint(100, 200), rng.randint(80, 150), rng.randint(80, 150))
        d.ellipse((x0, y0, x0 + r, y0 + r), fill=color)
    return im


def _lesion_left(im: Image.Image) -> Image.Image:
    d = ImageDraw.Draw(im)
    d.ellipse((6, 24, 22, 40), fill=(160, 40, 40))       # patient's left ≡ viewer's right per mirror
    d.ellipse((CANVAS - 16, 24, CANVAS - 8, 40), fill=(210, 190, 190))  # opposite side lighter
    return im


def _lesion_right(im: Image.Image) -> Image.Image:
    d = ImageDraw.Draw(im)
    d.ellipse((CANVAS - 22, 24, CANVAS - 6, 40), fill=(160, 40, 40))
    d.ellipse((8, 24, 16, 40), fill=(210, 190, 190))
    return im


def _lesion_midline(im: Image.Image) -> Image.Image:
    d = ImageDraw.Draw(im)
    d.rectangle((CANVAS // 2 - 6, 20, CANVAS // 2 + 6, 44), fill=(150, 40, 40))
    return im


def _closeup(im: Image.Image) -> Image.Image:
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, CANVAS, CANVAS), fill=(180, 150, 140))
    d.ellipse((CANVAS // 2 - 12, CANVAS // 2 - 12, CANVAS // 2 + 12, CANVAS // 2 + 12),
              fill=(140, 60, 60))  # isolated lesion, no orienting anatomy
    return im


def _corrupt_file(path: Path) -> None:
    path.write_bytes(b"NOT_AN_IMAGE_" + b"\x00" * 100)


def generate(dst: Path, *, n_per_class: int = 5) -> dict:
    """Populate ``dst/images`` and write ``dst/annotations.csv``."""
    img_dir = dst / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    idx = 0

    def _write(im: Image.Image, subdir: str, base: str, label: str) -> Path:
        nonlocal idx
        d = img_dir / subdir
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{base}_{idx:03d}.png"
        im.save(p)
        rows.append({"path": str(p.relative_to(img_dir)), "laterality": label})
        idx += 1
        return p

    for i in range(n_per_class):
        _write(_lesion_left(_base_image(1000 + i)), "left", "left", "LEFT")
        _write(_lesion_right(_base_image(2000 + i)), "right", "right", "RIGHT")
        _write(_lesion_midline(_base_image(3000 + i)), "midline", "mid", "BILATERAL_OR_MIDLINE")
        _write(_closeup(_base_image(4000 + i)), "closeup", "cu", "UNANSWERABLE")

    # Add exact duplicates for the dedup test
    if rows:
        src = img_dir / rows[0]["path"]
        for k in range(2):
            copy = src.with_name(f"exactdup_{k}.png")
            copy.write_bytes(src.read_bytes())
            rows.append({"path": str(copy.relative_to(img_dir)), "laterality": rows[0]["laterality"]})

    # Add a mirror of one left image → same source_group_id
    if rows:
        source = img_dir / rows[0]["path"]
        with Image.open(source) as im:
            mirror = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mp = source.with_name("mirror_of_left.png")
            mirror.save(mp)
            rows.append({"path": str(mp.relative_to(img_dir)), "laterality": "RIGHT"})

    # Add one corrupt file
    _corrupt_file(img_dir / "corrupt.jpg")
    rows.append({"path": "corrupt.jpg", "laterality": "EXCLUDE_CORRUPT_OR_NONCLINICAL"})

    # Add one unsupported file (not listed in annotations)
    (img_dir / "notes.txt").write_text("this is not an image\n")

    ann_path = dst / "annotations.csv"
    with ann_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "laterality"])
        w.writeheader()
        w.writerows(rows)

    return {
        "images_dir": img_dir,
        "annotations_csv": ann_path,
        "count": len(rows),
    }


if __name__ == "__main__":  # pragma: no cover
    import sys
    dst = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("samples")
    info = generate(dst)
    print(info)
