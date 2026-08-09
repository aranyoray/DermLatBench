#!/usr/bin/env python3
"""Explode Fitzpatrick17k parquet(s) -> PNG files under images/ for dermlat-audit."""
import io, sys, hashlib
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image

SRC = Path("/Users/raviraj/dermlat_data/parquet")
OUT = Path("/Users/raviraj/dermlat_data/images")
OUT.mkdir(parents=True, exist_ok=True)

def find_image_col(schema):
    # HF `image` feature -> struct<bytes: binary, path: string>, or raw binary
    for f in schema:
        t = str(f.type)
        if f.name.lower() in ("image", "img", "picture") or "bytes" in t:
            return f.name
    return schema.names[0]

def img_bytes(cell):
    if isinstance(cell, dict):
        return cell.get("bytes") or cell.get("image")
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    return None

total = 0
for pfile in sorted(SRC.glob("*.parquet")):
    tbl = pq.read_table(pfile)
    col = find_image_col(tbl.schema)
    print(f"{pfile.name}: rows={tbl.num_rows} image_col={col} cols={tbl.schema.names}", flush=True)
    data = tbl.column(col).to_pylist()
    for i, cell in enumerate(data):
        b = img_bytes(cell)
        if not b:
            continue
        try:
            im = Image.open(io.BytesIO(b)).convert("RGB")
        except Exception:
            continue
        h = hashlib.sha1(b).hexdigest()[:16]
        im.save(OUT / f"{pfile.stem}_{i:06d}_{h}.png")
        total += 1
        if total % 1000 == 0:
            print(f"  extracted {total}", flush=True)
print(f"EXTRACT_DONE total={total} -> {OUT}", flush=True)
