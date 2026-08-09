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
1. "laterality" refers to the PATIENT'S own left or right, not the viewer's. These are OPPOSITE in a
   front-facing photo: the patient's right side appears on the LEFT of the image. Never equate them.
2. THERE IS NO DEFAULT SIDE. LEFT and RIGHT are equally likely a priori. Do not prefer LEFT. When the
   evidence does not clearly single out one patient side, the answer is "UNANSWERABLE", not a guess.
3. To output LEFT or RIGHT, "visible_evidence" MUST name a PATIENT-INTRINSIC feature that distinguishes
   one side from the other — e.g. a midline structure with a known side (nose/philtrum/navel/spine and
   which side of it the lesion sits, accounting for whether the patient faces the camera), a clearly
   identifiable left-vs-right hand/foot from anatomical finger/toe order (NOT from where it sits in the
   frame), an ear whose shape identifies the side, or the contralateral limb being visible for reference.
4. FRAME POSITION IS NOT EVIDENCE. If your justification uses or would use "left/right side of the image",
   "on the left of the frame", "the thumb is on the left", or any position-within-the-picture cue, you MUST
   answer "UNANSWERABLE". A single close-up of one limb or region with no midline or bilateral reference is
   almost always "UNANSWERABLE".
5. SELF-CHECK before answering LEFT or RIGHT: (a) Does visible_evidence cite a patient-intrinsic landmark?
   (b) Did you correct for viewer-vs-patient mirroring? (c) Could a mirror-flip of this image change your
   answer? If (a) or (b) fails, or (c) is "yes", output "UNANSWERABLE".
6. You have no filename, folder name, metadata, or existing label. Do not invent any.
7. If the image is corrupt, non-clinical, or otherwise unusable, use "EXCLUDE_CORRUPT_OR_NONCLINICAL".
8. If two body sides are both visible and the lesion crosses midline or affects both, use "BILATERAL_OR_MIDLINE".
9. "confidence" must reflect genuine certainty about the PATIENT side. Cap it at 40 whenever the answer
   rested on any frame/position cue or a single side with no midline reference.
10. Set "review_required" to true when confidence is low or the answer required subtle judgment.
11. Set "possible_label_error" to true only if you would expect any reasonable annotator to disagree with a
    naive frame-based answer here.

Return the JSON object only. No preamble, no code fences, no trailing text.
"""


def prompt_text() -> str:
    return BLINDED_LATERALITY_PROMPT


def prompt_hash() -> str:
    return hashlib.sha256(BLINDED_LATERALITY_PROMPT.encode("utf-8")).hexdigest()
