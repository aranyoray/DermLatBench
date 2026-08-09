# DermLatBench audit report

_Generated_: `2026-08-09T21:14:41+00:00`
_Pipeline_: `dermlat-audit==0.1.0`, pipeline v`1.0.0`

## Methods
Every image under `paths.images_dir` was discovered, hashed (SHA-256), and its physical properties (dimensions, format, size, EXIF presence) recorded. Exact duplicates were grouped by SHA-256. Near-duplicates (mirrors, rotations, crops, resizes) were grouped by perceptual-hash bucketing plus per-bucket Hamming-distance comparison over eight canonical orientations, then reduced by union-find into a single `source_group_id` per equivalence class. Blinded local vision-language inference used the `ollama` backend (model `qwen2.5vl:7b`). The model received only image pixels and a fixed prompt — filenames, folders, EXIF, and supplied labels were withheld. Responses were parsed and validated against a fixed JSON schema; invalid or low-confidence responses received a second independent blinded pass. Disagreements between passes were flagged for human review. Two reviewers may independently re-label the flagged queue via a local Streamlit UI, with Reviewer 2 blinded to Reviewer 1's decision until adjudication. Metrics were computed with exact denominators; clustered bootstrap 95 % CIs use `source_group_id` as the cluster.

## Results

### Denominators
- total_discovered = **5,148**
- successfully_processed = **4,969**
- unprocessed = **179**
- corrupt = **0**
- unsupported = **0**

### Label distribution
| Label | Count | Rate |
|---|---:|---:|
| `LEFT` | 176 | 0.0354 |
| `RIGHT` | 57 | 0.0115 |
| `BILATERAL_OR_MIDLINE` | 371 | 0.0747 |
| `UNANSWERABLE` | 4365 | 0.8784 |
| `EXCLUDE_CORRUPT_OR_NONCLINICAL` | 0 | 0.0000 |

### Answerable / Unanswerable / Excluded
- answerable: **604** (0.1216)
- unanswerable: **4,365** (0.8784)
- excluded: **0** (0.0000)

### Supplied-label agreement
- n_comparable = **0**
- n_agree = **0**
- agreement = **0.0000**

#### Confusion matrix (supplied vs final)
*(no comparable rows — no supplied labels found)*

### Accuracy on answerable unilateral (supplied LEFT/RIGHT)
- n = **0**, correct = **0**
- accuracy = **0.0000**, macro_recall = **0.0000**, balanced_accuracy = **0.0000**

### Coverage / selective accuracy
- coverage = **0.0469**  selective_n = **0**  selective_accuracy = **0.0000**

### Abstention sensitivity / specificity
- sensitivity (abstain when supplied says UNANSWERABLE) = **n/a**
- specificity (answer when supplied says LEFT/RIGHT)     = **n/a**

### Confidence-stratified accuracy
| Bin | n | accuracy | coverage_of_bin |
|---|---:|---:|---:|
| 0-39 | 0 | n/a | 4373 |
| 40-59 | 0 | n/a | 48 |
| 60-79 | 0 | n/a | 59 |
| 80-100 | 0 | n/a | 489 |

### Reviewer agreement
- Cohen's κ = **n/a** (n = 0)
- Krippendorff's α = **n/a**

### Clustered bootstrap 95 % CI (accuracy on unilateral supplied labels)
- point = **0.0000**  95 % CI = **[0.0000, 0.0000]**  (n_iter = 0)

## Claim verification
| Claim | Claimed | Observed | Status | Notes |
|---|---:|---:|:-:|---|
| total_images | 12541 | 5148 | **CONTRADICTED** | observed=5148 vs claimed=12541 (diff=-7393) |
| answerable | 9770 | 604 | **CONTRADICTED** | observed=604 vs claimed=9770 (diff=-9166) |
| unanswerable | 2771 | 4365 | **CONTRADICTED** | observed=4365 vs claimed=2771 (diff=1594) |
| annotation_agreement>threshold | 0.9500 | n/a | **NOT_VERIFIABLE** | no observed value |

## Limitations
- The blinded model is not a clinical instrument. Every model output should be treated as a hypothesis for human review, not a diagnosis.
- Perceptual-hash near-duplicate grouping cannot detect all image manipulations (heavy recolouring, non-affine crops, semantic edits).
- Supplied labels may themselves be wrong; agreement statistics reflect concordance with the supplied labels, not with clinical truth.
- Reviewer agreement metrics require both reviewers to have covered the same rows.
- Clustered bootstrap CIs assume `source_group_id` is the only leakage boundary; patient-level linkage, if available separately, should be added as an additional cluster.

## Final statement
**Processed 4,969 of 5,148 discovered images; unprocessed 179; corrupt/excluded 0; human-review queue 4,632. Dataset-wide claims are [not ready] for publication.**