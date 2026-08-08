# Annotation guidelines

These rules apply to both the automated model and human reviewers.

## The five labels

| Label                              | Use when …                                                                    |
|------------------------------------|-------------------------------------------------------------------------------|
| `LEFT`                             | Visible orienting anatomy establishes the region is the **patient's** left.   |
| `RIGHT`                            | Same, but the patient's right.                                                |
| `BILATERAL_OR_MIDLINE`             | Both sides visible and involved, or the finding sits on the midline.          |
| `UNANSWERABLE`                     | You cannot honestly assign LEFT/RIGHT/BILATERAL from what the image shows.    |
| `EXCLUDE_CORRUPT_OR_NONCLINICAL`   | Image is corrupt, non-clinical (e.g. text screenshot), or otherwise unusable. |

## Reasoning rules

1. **"Left/Right" means the patient's own left or right, not the viewer's.**
   The most common mistake is flipping this.
2. **You must cite visible orienting anatomy.** Examples:
   - The contralateral limb is partly visible.
   - Midline landmarks (nose, umbilicus, spine) are visible.
   - Handedness signal (thumb position).
   - Ear morphology and orientation.
3. **Filename, folder, EXIF, and existing annotations don't count as evidence.**
   The blinded model doesn't see them. Reviewers should not use them either.
4. **Frame position ≠ patient laterality.** "The lesion is on the left of the image"
   is not evidence.
5. **Close-ups without orienting anatomy → `UNANSWERABLE`.** No guessing.
6. **When two sides are visible and both involved → `BILATERAL_OR_MIDLINE`.**
7. **`EXCLUDE_CORRUPT_OR_NONCLINICAL` is not a substitute for `UNANSWERABLE`.** Use it
   only when the file is unusable as an image, not merely uninformative.

## Answerability

`answerable = true` when you're providing LEFT, RIGHT, or BILATERAL_OR_MIDLINE.
`answerable = false` otherwise.

## Confidence

Integer 0–100. Reviewers: pick a number that means "if I saw 100 images like this, I'd
be right this many times". Low confidence (< 60) triggers a second pass; low confidence
should also usually set `review_required = true`.

## Possible label error

Set to `true` only when the visible evidence would lead any reasonable annotator to
disagree with a naive first pass. This is what surfaces suspected supplied-label errors
during audit.

## What the model sees

A single prompt (see `src/dermlat_audit/prompts.py::BLINDED_LATERALITY_PROMPT`) plus
the image pixels. Nothing else. The prompt hash is recorded per prediction so any change
to the wording is visible in the run manifest.
