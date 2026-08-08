from __future__ import annotations

import json

import pytest
from jsonschema import ValidationError

from dermlat_audit.schema import compute_flags, extract_json, validate_response


def _valid_obj(**over):
    d = {
        "laterality": "LEFT",
        "anatomical_site": "face",
        "answerable": True,
        "visible_evidence": "midline visible; nose to patient's left",
        "confidence": 80,
        "ambiguity_reason": None,
        "review_required": False,
        "possible_label_error": False,
    }
    d.update(over)
    return d


def test_extract_from_plain_json():
    obj = extract_json(json.dumps(_valid_obj()))
    validate_response(obj)


def test_extract_from_fenced_json():
    obj = extract_json("```json\n" + json.dumps(_valid_obj()) + "\n```")
    validate_response(obj)


def test_extract_from_noisy_response():
    text = "Sure! Here's my answer:\n" + json.dumps(_valid_obj()) + "\nThanks."
    obj = extract_json(text)
    validate_response(obj)


def test_malformed_json_raises():
    with pytest.raises(ValueError):
        extract_json("not json at all")


def test_missing_field_fails_schema():
    obj = _valid_obj()
    obj.pop("confidence")
    with pytest.raises(ValidationError):
        validate_response(obj)


def test_wrong_enum_fails_schema():
    obj = _valid_obj(laterality="MAYBE_LEFT")
    with pytest.raises(ValidationError):
        validate_response(obj)


def test_confidence_out_of_range_fails():
    obj = _valid_obj(confidence=999)
    with pytest.raises(ValidationError):
        validate_response(obj)


def test_flag_low_confidence():
    obj = _valid_obj(confidence=30)
    flags = compute_flags(obj, low_confidence_threshold=60)
    assert "low_confidence" in flags


def test_flag_frame_position_only():
    obj = _valid_obj(visible_evidence="Lesion appears on the left side of the image.")
    flags = compute_flags(obj, low_confidence_threshold=60)
    assert "frame_position_only" in flags


def test_flag_missing_evidence():
    obj = _valid_obj(visible_evidence="none")
    flags = compute_flags(obj, low_confidence_threshold=60)
    assert "missing_evidence" in flags


def test_flag_unsupported_certainty_answerable_unanswerable():
    obj = _valid_obj(laterality="UNANSWERABLE", answerable=True)
    flags = compute_flags(obj, low_confidence_threshold=60)
    assert "unsupported_certainty" in flags


def test_flag_possible_label_error():
    obj = _valid_obj(possible_label_error=True)
    flags = compute_flags(obj, low_confidence_threshold=60)
    assert "possible_label_error" in flags
