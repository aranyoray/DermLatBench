"""The single blinded prompt the model sees for every image.

RULES:
- Only image pixels and this exact text are shown to the model.
- The prompt never contains filename, folder, EXIF, or supplied labels.
- The prompt hash is recorded per prediction for reproducibility.
"""

from __future__ import annotations

import hashlib

BLINDED_LATERALITY_PROMPT = """You are auditing a dermatology image for anatomical laterality.

Return ONLY a JSON object matching this schema and NOTHING else:

{
  "laterality": "LEFT" | "RIGHT" | "BILATERAL_OR_MIDLINE" | "UNANSWERABLE" | "EXCLUDE_CORRUPT_OR_NONCLINICAL",
  "anatomical_site": string (best guess of body region) or null,
  "answerable": boolean,
  "visible_evidence": string describing the specific visible anatomical landmarks you used (or "none"),
  "confidence": integer 0-100,
  "ambiguity_reason": string or null,
  "review_required": boolean,
  "possible_label_error": boolean
}

STRICT REASONING RULES:
1. "laterality" refers to the PATIENT'S own left or right, not the viewer's.
2. You must justify LEFT or RIGHT by citing visible orienting anatomy the patient's body itself provides
   (e.g. midline landmarks, hair whorl, ear shape, thumb position, contralateral limb visible, etc.).
3. An isolated close-up lesion with no reliable orienting anatomy is "UNANSWERABLE". Do NOT guess.
4. You may NOT use frame position ("the lesion is on the left side of the image") as evidence of patient laterality.
5. You have no filename, folder name, metadata, or existing label. Do not invent any.
6. If the image is corrupt, non-clinical, or otherwise unusable, use "EXCLUDE_CORRUPT_OR_NONCLINICAL".
7. If two body sides are both visible and the lesion crosses midline or affects both, use "BILATERAL_OR_MIDLINE".
8. Set "review_required" to true when confidence is low or when the answer required a subtle judgment.
9. Set "possible_label_error" to true only if you would expect any reasonable annotator to disagree with a
   naive frame-based answer here.

Return the JSON object only. No preamble, no code fences, no trailing text.
"""


def prompt_text() -> str:
    return BLINDED_LATERALITY_PROMPT


def prompt_hash() -> str:
    return hashlib.sha256(BLINDED_LATERALITY_PROMPT.encode("utf-8")).hexdigest()
