from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw

from dermlat_audit.dedup import (
    ImageForDedup,
    compute_orientation_phashes,
    group_exact_duplicates,
    group_near_duplicates,
)
from dermlat_audit.hashing import phash_hex, sha256_file


def _mk(path: Path, seed: int = 0, base_color=(120, 60, 60)) -> Path:
    """Deterministically create a textured image so phashes are non-trivial."""
    im = Image.new("RGB", (64, 64), color=base_color)
    d = ImageDraw.Draw(im)
    rng = random.Random(seed)
    for _ in range(20):
        x = rng.randint(0, 63)
        y = rng.randint(0, 63)
        r = rng.randint(2, 10)
        c = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        d.ellipse((x, y, x + r, y + r), fill=c)
    im.save(path)
    return path


def test_exact_duplicate_groups(tmp_path: Path):
    a = _mk(tmp_path / "a.png", seed=1)
    b = tmp_path / "b.png"
    b.write_bytes(a.read_bytes())
    c = _mk(tmp_path / "c.png", seed=99, base_color=(30, 200, 30))

    items = [ImageForDedup(str(a), sha256_file(a), None),
             ImageForDedup(str(b), sha256_file(b), None),
             ImageForDedup(str(c), sha256_file(c), None)]
    groups = group_exact_duplicates(items)
    assert len(groups) == 1
    only_group = next(iter(groups.values()))
    assert set(only_group) == {str(a), str(b)}


def test_near_duplicate_groups_mirror(tmp_path: Path):
    a = _mk(tmp_path / "left.png", seed=42, base_color=(200, 40, 40))
    with Image.open(a) as im:
        im.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(tmp_path / "left_mirror.png")

    b = tmp_path / "left_mirror.png"
    unrelated = _mk(tmp_path / "other.png", seed=7777, base_color=(20, 20, 200))

    items = [
        ImageForDedup(str(a), sha256_file(a), phash_hex(a)),
        ImageForDedup(str(b), sha256_file(b), phash_hex(b)),
        ImageForDedup(str(unrelated), sha256_file(unrelated), phash_hex(unrelated)),
    ]
    orient = {
        str(a): compute_orientation_phashes(a),
        str(b): compute_orientation_phashes(b),
        str(unrelated): compute_orientation_phashes(unrelated),
    }
    groups = group_near_duplicates(items, orientation_hashes=orient,
                                   hamming_threshold=6, bucket_prefix_bits=8)
    # a and b should share a group; unrelated should be its own group.
    membership = {p: gid for gid, ps in groups.items() for p in ps}
    assert membership[str(a)] == membership[str(b)]
    assert membership[str(unrelated)] != membership[str(a)]


def test_every_readable_item_gets_a_group(tmp_path: Path):
    files = [_mk(tmp_path / f"{i}.png", seed=i * 100, base_color=(i * 20, 40, 40))
             for i in range(1, 6)]
    items = [ImageForDedup(str(f), sha256_file(f), phash_hex(f)) for f in files]
    groups = group_near_duplicates(items, hamming_threshold=6, bucket_prefix_bits=16)
    all_members = [p for ps in groups.values() for p in ps]
    assert set(all_members) == {str(f) for f in files}
