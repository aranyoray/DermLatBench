"""Duplicate & near-duplicate detection.

- Exact duplicates: group by SHA-256 (O(n)).
- Near duplicates: LSH-style bucketing on the top ``bucket_prefix_bits`` of the pHash of
  each of {original, mirror, 90°, 180°, 270°, mirror+90°, mirror+180°, mirror+270°}.
  Buckets are compared pairwise only within-bucket (typically <100 members), so overall
  cost is roughly O(n) for realistic dataset sizes.
- Groups are then reduced via union-find so that transitively-similar images share one
  ``source_group_id``.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .hashing import hamming_hex
from .utils import utc_now_iso


@dataclass(slots=True)
class ImageForDedup:
    path: str
    sha256: str | None
    phash: str | None


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent: dict[str, str] = {i: i for i in items}
        self.rank: dict[str, int] = dict.fromkeys(self.parent, 0)

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for item in self.parent:
            out[self.find(item)].append(item)
        return out


def compute_orientation_phashes(path: Path, hash_size: int = 8) -> list[str] | None:
    """Return pHashes for 8 canonical orientations (4 rotations x 2 flips)."""
    try:
        import imagehash
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("imagehash is required") from exc
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            variants = [
                im,
                im.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
                im.rotate(90, expand=True),
                im.rotate(180, expand=True),
                im.rotate(270, expand=True),
                im.transpose(Image.Transpose.FLIP_LEFT_RIGHT).rotate(90, expand=True),
                im.transpose(Image.Transpose.FLIP_LEFT_RIGHT).rotate(180, expand=True),
                im.transpose(Image.Transpose.FLIP_LEFT_RIGHT).rotate(270, expand=True),
            ]
            return [str(imagehash.phash(v, hash_size=hash_size)) for v in variants]
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def _bucket_key(phash_hex: str, prefix_bits: int) -> str:
    """Take the top ``prefix_bits`` of a hex hash as the bucket key."""
    total_bits = len(phash_hex) * 4
    if prefix_bits > total_bits:
        prefix_bits = total_bits
    # Convert to a fixed-width binary string, take prefix.
    n_hex = prefix_bits // 4
    return phash_hex[:n_hex] if n_hex > 0 else "_"


def group_exact_duplicates(items: list[ImageForDedup]) -> dict[str, list[str]]:
    """Return {duplicate_group_id -> [paths]} for SHA-256 collisions with >=2 members."""
    by_sha: dict[str, list[str]] = defaultdict(list)
    for it in items:
        if it.sha256:
            by_sha[it.sha256].append(it.path)
    return {f"dup_{sha[:12]}": paths for sha, paths in by_sha.items() if len(paths) >= 2}


def group_near_duplicates(
    items: list[ImageForDedup],
    *,
    orientation_hashes: dict[str, list[str]] | None = None,
    hamming_threshold: int = 6,
    bucket_prefix_bits: int = 16,
) -> dict[str, list[str]]:
    """Return {source_group_id -> [paths]}.

    Every readable image ends up in exactly one group (singletons included).

    Parameters
    ----------
    items
        All images to cluster. Members without a phash are placed in singleton groups.
    orientation_hashes
        Optional {path -> [8 pHashes]} for rotation/mirror-aware matching.
        Falls back to a single phash from ``item.phash`` if not provided.
    """
    uf = UnionFind(it.path for it in items)

    # Bucket every path by every one of its orientation-hashes' prefix.
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)  # bucket -> [(path, phash)]
    for it in items:
        hashes: list[str] = []
        if orientation_hashes and it.path in orientation_hashes and orientation_hashes[it.path]:
            hashes = orientation_hashes[it.path]
        elif it.phash:
            hashes = [it.phash]
        for h in hashes:
            buckets[_bucket_key(h, bucket_prefix_bits)].append((it.path, h))

    # Within each bucket, do pairwise comparisons (small).
    for members in buckets.values():
        n = len(members)
        if n < 2:
            continue
        for i in range(n):
            pi, hi = members[i]
            for j in range(i + 1, n):
                pj, hj = members[j]
                if pi == pj:
                    continue
                try:
                    d = hamming_hex(hi, hj)
                except ValueError:
                    continue
                if d <= hamming_threshold:
                    uf.union(pi, pj)

    raw_groups = uf.groups()
    return {f"src_{uuid.uuid5(uuid.NAMESPACE_URL, root).hex[:12]}": sorted(members)
            for root, members in raw_groups.items()}


def summarize_groups(groups: dict[str, list[str]]) -> dict[str, int | str]:
    sizes = [len(v) for v in groups.values()]
    return {
        "total_groups": len(groups),
        "singletons": sum(1 for s in sizes if s == 1),
        "multi_member_groups": sum(1 for s in sizes if s >= 2),
        "max_group_size": max(sizes) if sizes else 0,
        "created_at": utc_now_iso(),
    }
