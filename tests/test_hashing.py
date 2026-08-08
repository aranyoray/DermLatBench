from __future__ import annotations

from pathlib import Path

from PIL import Image

from dermlat_audit.hashing import hamming_hex, phash_hex, sha256_file


def test_sha256_reproducible(tmp_path: Path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello world")
    assert sha256_file(p) == sha256_file(p)
    q = tmp_path / "b.bin"
    q.write_bytes(b"hello world")
    assert sha256_file(p) == sha256_file(q)


def test_phash_returns_hex(tmp_path: Path):
    p = tmp_path / "im.png"
    Image.new("RGB", (32, 32), color=(120, 60, 60)).save(p)
    h = phash_hex(p)
    assert isinstance(h, str) and len(h) > 0
    int(h, 16)  # parses as hex


def test_phash_none_on_bad_file(tmp_path: Path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"not an image")
    assert phash_hex(p) is None


def test_hamming_hex_basic():
    assert hamming_hex("00", "00") == 0
    assert hamming_hex("ff", "00") == 8
    assert hamming_hex("aa", "55") == 8


def test_phash_near_duplicates_have_small_distance(tmp_path: Path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    im = Image.new("RGB", (64, 64), color=(128, 100, 100))
    im.save(a)
    # Add a tiny difference so files aren't byte-identical.
    im2 = im.copy()
    im2.putpixel((0, 0), (129, 100, 100))
    im2.save(b)
    assert hamming_hex(phash_hex(a), phash_hex(b)) < 6
