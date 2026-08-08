"""Fixed labels and enumerations. Do not change without a schema-migration bump."""

from __future__ import annotations

from enum import Enum

# The five mutually-exclusive laterality labels.
LABEL_LEFT = "LEFT"
LABEL_RIGHT = "RIGHT"
LABEL_BILATERAL_OR_MIDLINE = "BILATERAL_OR_MIDLINE"
LABEL_UNANSWERABLE = "UNANSWERABLE"
LABEL_EXCLUDE = "EXCLUDE_CORRUPT_OR_NONCLINICAL"

ALL_LABELS: tuple[str, ...] = (
    LABEL_LEFT,
    LABEL_RIGHT,
    LABEL_BILATERAL_OR_MIDLINE,
    LABEL_UNANSWERABLE,
    LABEL_EXCLUDE,
)

# Labels that count as "answerable unilateral" for accuracy metrics.
UNILATERAL_LABELS: tuple[str, ...] = (LABEL_LEFT, LABEL_RIGHT)

# Labels that count as an abstention from a unilateral call.
ABSTENTION_LABELS: tuple[str, ...] = (
    LABEL_BILATERAL_OR_MIDLINE,
    LABEL_UNANSWERABLE,
    LABEL_EXCLUDE,
)


class ReviewFlag(str, Enum):
    """Reasons a prediction was routed to human review."""

    LOW_CONFIDENCE = "low_confidence"
    DISAGREEMENT_BETWEEN_PASSES = "disagreement_between_passes"
    INVALID_JSON = "invalid_json"
    MISSING_EVIDENCE = "missing_evidence"
    FRAME_POSITION_ONLY = "frame_position_only"
    UNSUPPORTED_CERTAINTY = "unsupported_certainty"
    POSSIBLE_LABEL_ERROR = "possible_label_error"
    MODEL_ERROR = "model_error"


# Supported image extensions (lower-case, with leading dot).
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
    ".gif", ".webp", ".heic", ".heif",
})

# JSON Schema for a single model response (validated with jsonschema).
MODEL_RESPONSE_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "laterality",
        "answerable",
        "visible_evidence",
        "confidence",
        "review_required",
        "possible_label_error",
    ],
    "properties": {
        "laterality": {"type": "string", "enum": list(ALL_LABELS)},
        "anatomical_site": {"type": ["string", "null"], "maxLength": 200},
        "answerable": {"type": "boolean"},
        "visible_evidence": {"type": "string", "minLength": 1, "maxLength": 2000},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "ambiguity_reason": {"type": ["string", "null"], "maxLength": 500},
        "review_required": {"type": "boolean"},
        "possible_label_error": {"type": "boolean"},
    },
}
