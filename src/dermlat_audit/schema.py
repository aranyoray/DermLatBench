"""Robust extraction + validation of the model's JSON output."""

from __future__ import annotations

import json
import re
from typing import Any

import jsonschema

from .constants import MODEL_RESPONSE_SCHEMA, ReviewFlag

_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_FIRST_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_FRAME_POSITION_TERMS = (
    "left side of the image",
    "right side of the image",
    "left of the frame",
    "right of the frame",
    "on the left of the photo",
    "on the right of the photo",
    "in the left part of the image",
    "in the right part of the image",
    "left half of the image",
    "right half of the image",
)


def extract_json(text: str) -> dict[str, Any]:
    """Best-effort recovery of a JSON object from a model completion.

    Raises ValueError if no plausible JSON object can be recovered.
    """
    if text is None:
        raise ValueError("empty response")
    cleaned = _CODE_FENCE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = _FIRST_OBJECT.search(cleaned)
    if not m:
        raise ValueError("no JSON object in response")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON object: {exc}") from exc


def validate_response(obj: dict[str, Any]) -> None:
    """Raise jsonschema.ValidationError if ``obj`` does not match the response schema."""
    jsonschema.validate(instance=obj, schema=MODEL_RESPONSE_SCHEMA)


def compute_flags(obj: dict[str, Any], low_confidence_threshold: int) -> list[str]:
    """Semantic flags on top of schema validity."""
    flags: list[str] = []
    conf = obj.get("confidence")
    if isinstance(conf, int) and conf < low_confidence_threshold:
        flags.append(ReviewFlag.LOW_CONFIDENCE.value)
    evidence = (obj.get("visible_evidence") or "").strip().lower()
    if evidence in ("", "none", "n/a", "na"):
        flags.append(ReviewFlag.MISSING_EVIDENCE.value)
    if any(term in evidence for term in _FRAME_POSITION_TERMS):
        flags.append(ReviewFlag.FRAME_POSITION_ONLY.value)
    lat = obj.get("laterality")
    if obj.get("answerable") is True and lat == "UNANSWERABLE":
        flags.append(ReviewFlag.UNSUPPORTED_CERTAINTY.value)
    if obj.get("answerable") is False and lat in ("LEFT", "RIGHT"):
        flags.append(ReviewFlag.UNSUPPORTED_CERTAINTY.value)
    if obj.get("review_required") is True:
        flags.append(ReviewFlag.LOW_CONFIDENCE.value)
    if obj.get("possible_label_error") is True:
        flags.append(ReviewFlag.POSSIBLE_LABEL_ERROR.value)
    return sorted(set(flags))
