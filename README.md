# dermlat-audit

Independent, reproducible audit pipeline for the **DermLatBench** dermatology laterality
dataset. Runs entirely on-device on Apple Silicon. **Makes no cloud or paid AI calls
during batch inference.**

> **⚠️ Not a medical device.** This is a research auditing tool. Do not treat any output
> as clinical ground truth.

## What it does

1. **Preflight & inventory** — recursively discovers images, computes SHA-256 / dimensions /
   format / size / EXIF presence, records everything in a SQLite manifest, and reconciles the
   true image count against manuscript claims.
2. **Duplicate detection** — exact SHA-256 duplicates plus scalable perceptual-hash near-
   duplicate detection (bucketed, avoids O(n²)). Groups mirrors / rotations / crops / resizes
   under one `source_group_id`.
3. **Blinded local inference** — runs an on-device vision-language model (default: MLX-VLM
   Qwen2-VL) against a standardized laterality prompt. Filenames, folders, EXIF, and any
   supplied labels are stripped before the model sees the image.
4. **Reliability** — second independent blinded pass on low-confidence cases; flags for
   disagreement, invalid JSON, missing evidence, and answers that depend only on frame
   position.
5. **Human review** — Streamlit app with two blinded reviewers plus an adjudication mode.
6. **Statistical analysis** — accuracy, macro recall, balanced accuracy, coverage, selective
   accuracy, abstention sensitivity/specificity, confidence-stratified breakdowns, Cohen's κ,
   Krippendorff's α, source-group–clustered bootstrap 95 % CIs.
7. **Leakage-safe splits** — train/val/test by `source_group_id`; fails validation if any
   overlap is detected.
8. **Claim verification** — labels each manuscript claim (12,541 total / 9,770 answerable /
   2,771 unanswerable / >95 % agreement) as `VERIFIED`, `CONTRADICTED`, or `NOT_VERIFIABLE`.
9. **Reproducibility** — SQLite is transactional; all runs are idempotent; every config,
   prompt, model id, hash, and software version is recorded to a run manifest.

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4). Intel Macs and Linux work for the CPU-only mock
  backend, useful for tests but too slow for 12k images.
- Python 3.11 or newer.
- 16 GB RAM minimum (a 2B 4-bit VLM fits comfortably). 24 GB+ recommended for 7B models.
- ~5 GB free disk for a small MLX model, plus space for outputs (~1–2 GB for 12k images).

## Quick start

```bash
git clone https://github.com/CHANGE_ME/dermlat-audit.git
cd dermlat-audit

./scripts/setup_mac.sh              # detects hardware, installs deps

./scripts/run_audit.sh \
  --images "/path/to/images" \
  --annotations "/path/to/annotations.csv" \
  --output "/path/to/private-results"

# If it gets interrupted, resume:
./scripts/resume_audit.sh --output "/path/to/private-results"

# Two-reviewer web UI:
dermlat-audit review --output "/path/to/private-results"

# Regenerate report from an existing DB:
./scripts/generate_report.sh --output "/path/to/private-results"
```

While running, the CLI prints:

```
processed 4,213 / 12,541   [========>          ]  33.6 %
rate: 2.4 img/s   ETA: 57m 12s
flagged: 812    failures: 3
model: mlx-community/Qwen2-VL-2B-Instruct-4bit
resume with: dermlat-audit resume --output /path/to/private-results
```

## Outputs

Everything lands under `--output`:

| File                          | What                                                             |
|-------------------------------|------------------------------------------------------------------|
| `complete_manifest.csv`       | Every discovered file with hash / size / format / EXIF flag.     |
| `automated_predictions.csv`   | Per-image model output (validated JSON, blinded).                |
| `flagged_human_review.csv`    | Queue for the reviewer UI.                                       |
| `reviewer_1.csv`              | Reviewer 1 independent decisions.                                |
| `reviewer_2.csv`              | Reviewer 2 independent decisions (blinded to Reviewer 1).        |
| `adjudicated_labels.csv`      | Post-adjudication decisions.                                     |
| `duplicate_groups.csv`        | Exact-duplicate SHA-256 groups.                                  |
| `source_groups.csv`           | Near-duplicate / mirror / crop / rotation groups.                |
| `leakage_safe_splits.csv`     | train/val/test assignment by `source_group_id`.                  |
| `metrics.json`                | All statistics with exact denominators + 95 % CIs.               |
| `audit_report.md`             | Human-readable report incl. Methods, Results, Limitations.       |
| `run_manifest.json`           | Machine-readable manifest of config, prompts, models, versions.  |
| `figures/`, `tables/`         | Paper-ready assets.                                              |

## Classification schema

Every image ends up in exactly one of:

- `LEFT`
- `RIGHT`
- `BILATERAL_OR_MIDLINE`
- `UNANSWERABLE`
- `EXCLUDE_CORRUPT_OR_NONCLINICAL`

Reasoning rules the model and reviewers must follow:

- An isolated lesion close-up without orienting anatomy is `UNANSWERABLE`.
- Filename, folder, metadata, frame position, and existing annotations do **not** count as
  visual evidence.
- Patient laterality is the **patient's** left/right, not the viewer's.

See [`docs/ANNOTATION_GUIDELINES.md`](docs/ANNOTATION_GUIDELINES.md) for the full rubric.

## Backends

Interchangeable via `--backend`:

| Backend  | When to use                                              | Requires                     |
|----------|----------------------------------------------------------|------------------------------|
| `mlx`    | **Default on Apple Silicon.** Fastest.                   | `pip install -e .[mlx]`      |
| `ollama` | Fallback / cross-platform / larger models.               | `ollama serve` + a vision model |
| `mock`   | Tests, smoke, CI. Deterministic, offline.                | Nothing extra.               |

## Privacy

- Images are never uploaded anywhere.
- The default `.gitignore` refuses to track images, annotations, outputs, DBs, and model
  files. `dermlat-audit doctor` prints the exact files that would be committed if you ran
  `git add -A`.

## Development

```bash
pip install -e ".[all,mlx]"
ruff check .
mypy src
pytest -q
```

## Repository layout

```
src/dermlat_audit/       # library + CLI + Streamlit UI
tests/                   # pytest suite
tests/synthetic/         # generator for synthetic no-PHI test images
scripts/                 # user-facing bash scripts
docs/                    # ARCHITECTURE / ANNOTATION_GUIDELINES / TROUBLESHOOTING
samples/                 # generated synthetic samples (safe to commit)
```

## License

MIT. See [`LICENSE`](LICENSE) and [`CITATION.cff`](CITATION.cff).
