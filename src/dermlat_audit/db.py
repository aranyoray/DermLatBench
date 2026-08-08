"""SQLite storage. Transactional; idempotent inserts; safe to interrupt.

Schema (v1):
    images             one row per discovered file
    predictions        one row per (image, model, pass_number)
    review_queue       one row per image needing human review
    reviews            one row per (image, reviewer_id)
    adjudications      one row per image after both reviewers
    source_groups      one row per near-duplicate group
    duplicate_groups   one row per exact-SHA group
    run_meta           key/value store for run manifest + versions
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .manifest import ManifestRow

SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS images (
    path TEXT PRIMARY KEY,
    relpath TEXT NOT NULL,
    sha256 TEXT,
    size_bytes INTEGER NOT NULL,
    width INTEGER,
    height INTEGER,
    format TEXT,
    mode TEXT,
    exif_present INTEGER NOT NULL,
    supported INTEGER NOT NULL,
    readable INTEGER NOT NULL,
    error TEXT,
    discovered_at TEXT NOT NULL,
    phash TEXT,
    source_group_id TEXT,
    duplicate_group_id TEXT,
    supplied_label TEXT
);
CREATE INDEX IF NOT EXISTS idx_images_sha256 ON images(sha256);
CREATE INDEX IF NOT EXISTS idx_images_source_group ON images(source_group_id);

CREATE TABLE IF NOT EXISTS predictions (
    path TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    pass_number INTEGER NOT NULL,
    pipeline_version TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    parsed_json TEXT,
    laterality TEXT,
    answerable INTEGER,
    confidence INTEGER,
    anatomical_site TEXT,
    visible_evidence TEXT,
    ambiguity_reason TEXT,
    review_required INTEGER,
    possible_label_error INTEGER,
    flags TEXT,
    valid INTEGER NOT NULL,
    error TEXT,
    processed_at TEXT NOT NULL,
    duration_ms INTEGER,
    PRIMARY KEY (path, model_name, pass_number),
    FOREIGN KEY (path) REFERENCES images(path)
);

CREATE TABLE IF NOT EXISTS review_queue (
    path TEXT PRIMARY KEY,
    reasons TEXT NOT NULL,
    added_at TEXT NOT NULL,
    FOREIGN KEY (path) REFERENCES images(path)
);

CREATE TABLE IF NOT EXISTS reviews (
    path TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    laterality TEXT NOT NULL,
    answerable INTEGER NOT NULL,
    anatomical_site TEXT,
    visible_evidence TEXT,
    confidence INTEGER,
    notes TEXT,
    reviewed_at TEXT NOT NULL,
    PRIMARY KEY (path, reviewer_id),
    FOREIGN KEY (path) REFERENCES images(path)
);

CREATE TABLE IF NOT EXISTS adjudications (
    path TEXT PRIMARY KEY,
    laterality TEXT NOT NULL,
    answerable INTEGER NOT NULL,
    anatomical_site TEXT,
    visible_evidence TEXT,
    adjudicator_id TEXT NOT NULL,
    notes TEXT,
    adjudicated_at TEXT NOT NULL,
    FOREIGN KEY (path) REFERENCES images(path)
);

CREATE TABLE IF NOT EXISTS source_groups (
    source_group_id TEXT PRIMARY KEY,
    member_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS duplicate_groups (
    duplicate_group_id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    member_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Database:
    """Thin wrapper around sqlite3 with sensible defaults."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.path),
            timeout=30.0,
            isolation_level=None,          # manual transaction control
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(_SCHEMA_SQL)
        cur.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        cur = self._conn.cursor()
        cur.execute("BEGIN IMMEDIATE;")
        try:
            yield cur
        except Exception:
            cur.execute("ROLLBACK;")
            raise
        else:
            cur.execute("COMMIT;")

    # ---------- images ----------
    def upsert_manifest_rows(self, rows: Iterable[ManifestRow]) -> int:
        rows = list(rows)
        with self.transaction() as cur:
            for r in rows:
                cur.execute(
                    """
                    INSERT INTO images(
                        path, relpath, sha256, size_bytes, width, height, format, mode,
                        exif_present, supported, readable, error, discovered_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        relpath=excluded.relpath,
                        sha256=excluded.sha256,
                        size_bytes=excluded.size_bytes,
                        width=excluded.width,
                        height=excluded.height,
                        format=excluded.format,
                        mode=excluded.mode,
                        exif_present=excluded.exif_present,
                        supported=excluded.supported,
                        readable=excluded.readable,
                        error=excluded.error,
                        discovered_at=excluded.discovered_at
                    """,
                    (
                        r.path, r.relpath, r.sha256, r.size_bytes, r.width, r.height,
                        r.format, r.mode, int(r.exif_present), int(r.supported),
                        int(r.readable), r.error, r.discovered_at,
                    ),
                )
        return len(rows)

    def set_phash(self, path: str, phash: str | None) -> None:
        self._conn.execute("UPDATE images SET phash=? WHERE path=?", (phash, path))

    def set_source_group(self, path: str, group_id: str) -> None:
        self._conn.execute(
            "UPDATE images SET source_group_id=? WHERE path=?", (group_id, path)
        )

    def set_duplicate_group(self, path: str, group_id: str) -> None:
        self._conn.execute(
            "UPDATE images SET duplicate_group_id=? WHERE path=?", (group_id, path)
        )

    def set_supplied_label(self, path: str, label: str | None) -> None:
        self._conn.execute("UPDATE images SET supplied_label=? WHERE path=?", (label, path))

    def all_images(self) -> list[sqlite3.Row]:
        return list(self._conn.execute("SELECT * FROM images ORDER BY path"))

    def readable_images_without_predictions(self, model_name: str) -> list[sqlite3.Row]:
        cur = self._conn.execute(
            """
            SELECT i.* FROM images i
            LEFT JOIN predictions p
                ON p.path = i.path AND p.model_name = ? AND p.pass_number = 1
            WHERE i.readable = 1 AND p.path IS NULL
            ORDER BY i.path
            """,
            (model_name,),
        )
        return list(cur)

    def images_needing_second_pass(self, model_name: str, threshold: int) -> list[sqlite3.Row]:
        cur = self._conn.execute(
            """
            SELECT i.* FROM images i
            JOIN predictions p1 ON p1.path = i.path AND p1.model_name = ? AND p1.pass_number = 1
            LEFT JOIN predictions p2 ON p2.path = i.path AND p2.model_name = ? AND p2.pass_number = 2
            WHERE i.readable = 1
              AND p2.path IS NULL
              AND (p1.valid = 0 OR p1.confidence IS NULL OR p1.confidence < ?)
            ORDER BY i.path
            """,
            (model_name, model_name, threshold),
        )
        return list(cur)

    # ---------- predictions ----------
    def insert_prediction(self, row: dict[str, Any]) -> None:
        cols = list(row.keys())
        placeholders = ",".join(["?"] * len(cols))
        col_sql = ",".join(cols)
        with self.transaction() as cur:
            cur.execute(
                f"INSERT OR REPLACE INTO predictions({col_sql}) VALUES ({placeholders})",
                tuple(row[c] for c in cols),
            )

    def predictions_for(self, path: str) -> list[sqlite3.Row]:
        return list(self._conn.execute(
            "SELECT * FROM predictions WHERE path=? ORDER BY pass_number", (path,)
        ))

    def all_first_pass_predictions(self, model_name: str) -> list[sqlite3.Row]:
        return list(self._conn.execute(
            "SELECT * FROM predictions WHERE model_name=? AND pass_number=1", (model_name,)
        ))

    def all_predictions(self) -> list[sqlite3.Row]:
        return list(self._conn.execute("SELECT * FROM predictions ORDER BY path, pass_number"))

    # ---------- review ----------
    def add_to_review_queue(self, path: str, reasons: list[str], added_at: str) -> None:
        self._conn.execute(
            """
            INSERT INTO review_queue(path, reasons, added_at) VALUES(?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                reasons=excluded.reasons,
                added_at=excluded.added_at
            """,
            (path, json.dumps(sorted(set(reasons))), added_at),
        )

    def review_queue(self) -> list[sqlite3.Row]:
        return list(self._conn.execute(
            """
            SELECT rq.path, rq.reasons, rq.added_at, i.*
            FROM review_queue rq JOIN images i ON i.path = rq.path
            ORDER BY rq.added_at
            """
        ))

    def upsert_review(self, review: dict[str, Any]) -> None:
        cols = list(review.keys())
        placeholders = ",".join(["?"] * len(cols))
        col_sql = ",".join(cols)
        update_sql = ",".join(f"{c}=excluded.{c}" for c in cols if c not in ("path", "reviewer_id"))
        with self.transaction() as cur:
            cur.execute(
                f"INSERT INTO reviews({col_sql}) VALUES ({placeholders}) "
                f"ON CONFLICT(path, reviewer_id) DO UPDATE SET {update_sql}",
                tuple(review[c] for c in cols),
            )

    def reviews_by_reviewer(self, reviewer_id: str) -> list[sqlite3.Row]:
        return list(self._conn.execute(
            "SELECT * FROM reviews WHERE reviewer_id=? ORDER BY path", (reviewer_id,)
        ))

    def all_reviews(self) -> list[sqlite3.Row]:
        return list(self._conn.execute("SELECT * FROM reviews ORDER BY path, reviewer_id"))

    def upsert_adjudication(self, adj: dict[str, Any]) -> None:
        cols = list(adj.keys())
        placeholders = ",".join(["?"] * len(cols))
        col_sql = ",".join(cols)
        update_sql = ",".join(f"{c}=excluded.{c}" for c in cols if c != "path")
        with self.transaction() as cur:
            cur.execute(
                f"INSERT INTO adjudications({col_sql}) VALUES ({placeholders}) "
                f"ON CONFLICT(path) DO UPDATE SET {update_sql}",
                tuple(adj[c] for c in cols),
            )

    def all_adjudications(self) -> list[sqlite3.Row]:
        return list(self._conn.execute("SELECT * FROM adjudications ORDER BY path"))

    # ---------- run meta ----------
    def set_meta(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO run_meta(key, value) VALUES (?, ?)",
            (key, json.dumps(value, default=str)),
        )

    def get_meta(self, key: str) -> Any:
        row = self._conn.execute("SELECT value FROM run_meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else None

    def all_meta(self) -> dict[str, Any]:
        rows = self._conn.execute("SELECT key, value FROM run_meta")
        return {r["key"]: json.loads(r["value"]) for r in rows}


def manifest_row_dict(r: ManifestRow) -> dict[str, Any]:
    return asdict(r)
