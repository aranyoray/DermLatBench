from __future__ import annotations

from pathlib import Path

from dermlat_audit.db import Database
from dermlat_audit.manifest import discover_images


def test_manifest_discovers_all_files(synthetic_dataset):
    rows = discover_images(synthetic_dataset["images_dir"])
    # Every file — even the corrupt jpg and the notes.txt — is present.
    names = {Path(r.path).name for r in rows}
    assert "corrupt.jpg" in names
    assert "notes.txt" in names
    assert any(r.readable for r in rows)


def test_manifest_flags_corrupt_and_unsupported(synthetic_dataset):
    rows = discover_images(synthetic_dataset["images_dir"])
    by_name = {Path(r.path).name: r for r in rows}
    assert not by_name["corrupt.jpg"].readable
    assert by_name["corrupt.jpg"].supported
    assert not by_name["notes.txt"].supported


def test_db_upsert_and_query(tmp_path: Path, synthetic_dataset):
    rows = discover_images(synthetic_dataset["images_dir"])
    with Database(tmp_path / "audit.sqlite") as db:
        db.upsert_manifest_rows(rows)
        again = db.upsert_manifest_rows(rows)  # idempotent
        assert again == len(rows)
        stored = db.all_images()
        assert len(stored) == len(rows)


def test_transactional_rollback(tmp_path: Path):
    db = Database(tmp_path / "audit.sqlite")
    try:
        with db.transaction() as cur:
            cur.execute(
                "INSERT INTO images(path,relpath,size_bytes,exif_present,supported,readable,discovered_at) "
                "VALUES('/a','a',10,0,1,1,'t')"
            )
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert db.all_images() == []
    db.close()
