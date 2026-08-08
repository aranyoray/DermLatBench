# Architecture

## Pipeline stages

```
┌─────────────┐    ┌─────────────┐    ┌────────────────────┐    ┌──────────────┐
│  preflight  │ →  │    dedup    │ →  │ blinded inference  │ →  │   report     │
│ (inventory) │    │  (source_   │    │  (VLM, pass 1+2)   │    │ (CSV/MD/JSON)│
└─────────────┘    │  group_id)  │    └────────────────────┘    └──────────────┘
                   └─────────────┘             │
                                               ▼
                                      ┌──────────────────┐
                                      │ human review UI  │
                                      │ (Streamlit)      │
                                      └──────────────────┘
```

Every stage writes to a single SQLite database (`audit.sqlite` under the output dir) via
`IMMEDIATE` transactions. That gives us:

- **Idempotence.** Preflight upserts by absolute path. Dedup upserts by group id.
  Inference has a `PRIMARY KEY (path, model_name, pass_number)` so re-running is a no-op
  on already-processed rows.
- **Safe interruption.** SIGKILL between images loses at most the current batch's
  transaction; the next run picks up where it stopped.
- **Reproducibility.** Every run stamps its config, prompt hash, model name/version,
  hardware, and software versions into the `run_meta` table.

## Data flow

1. **Preflight** — recursively walks `paths.images_dir`, opens each file with PIL, records
   `sha256`, dimensions, format, size, EXIF flag, and per-file readability. Nothing is
   silently skipped; unsupported and unreadable files are still rows in `images`.
2. **Dedup** — computes a perceptual hash for each readable file across 8 canonical
   orientations (4 rotations × 2 mirror axes), buckets by hash prefix, then does within-
   bucket Hamming-distance comparisons. Union-find reduces transitively similar images
   into one `source_group_id`. Exact-SHA-256 collisions form the `duplicate_group_id`.
3. **Inference pass 1** — for every readable image without a pass-1 prediction, run the
   backend with only pixels + fixed prompt. Parse response, validate against
   `MODEL_RESPONSE_SCHEMA`, compute semantic flags, upsert into `predictions`. Flagged
   rows enter `review_queue`.
4. **Inference pass 2** — same, but only for rows where pass 1 was invalid OR
   `confidence < low_confidence_threshold`.
5. **Disagreement flag** — for images with both passes, if the label differs, add
   `disagreement_between_passes` to `review_queue.reasons`.
6. **Report** — emit all CSVs, metrics.json, audit_report.md, run_manifest.json, plus
   `tables/` and `figures/`.

## Blinding

The model backend interface is deliberately narrow:

```python
def infer(self, image_path: Path, prompt: str, *, timeout_seconds: int) -> InferenceResult
```

No supplied labels, filenames, EXIF, or annotation metadata can be threaded through — the
`test_blinding.py` suite asserts this at the type-signature level and at the pipeline
call level.

## Leakage-safe splits

Splits are assigned by hashing `source_group_id`, not `path`. All derivatives of one
source image share a group, so they always land in the same split. `validate_no_leakage`
re-checks this invariant and raises `LeakageError` on any overlap; the report bakes the
result into `metrics.leakage_check.ok`.

## SQLite schema (v1)

See `db.py` for the authoritative DDL. Key tables:

| Table              | Row = one …                                              |
|--------------------|----------------------------------------------------------|
| `images`           | discovered file (readable or not)                        |
| `predictions`      | (image × model × pass) blinded model output              |
| `review_queue`     | image needing a human reviewer                           |
| `reviews`          | (image × reviewer_id) independent human decision         |
| `adjudications`    | image, after both reviewers, adjudicated                 |
| `source_groups`    | one near-duplicate cluster                               |
| `duplicate_groups` | one exact-SHA-256 cluster                                |
| `run_meta`         | free-form key/value store for the run manifest           |

## Extending backends

Implement `Backend`, register it in `inference/base.py::make_backend`, and update
`config.py` to accept the new literal in the `backend:` field.

## Where things get written

Every artefact lives under `paths.output_dir`. Nothing is written to the images
directory, and nothing is written to the repo checkout. The default `.gitignore` refuses
to track any of these paths.
