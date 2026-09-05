from __future__ import annotations

import math

import pytest

from astroswarm_mcp.algorithm import (
    EXAMPLE_ALGORITHM,
    generate_ring_placements,
    normalize_algorithm,
    validate_algorithm,
    validate_placements,
)


def test_example_algorithm_is_valid() -> None:
    result = validate_algorithm(EXAMPLE_ALGORITHM)

    assert result.valid
    assert result.errors == []
    assert result.as_dict()["block_count"] == 11
    assert result.as_dict()["effective_config"]["speed_m_s"] == 5.25


def test_flat_legacy_algorithm_is_normalized_like_godot() -> None:
    flat = [
        {"type": "set_speed", "params": {"value": 4.5}},
        {"type": "when_always", "params": {}},
        {"type": "do_forward", "params": {}},
    ]

    normalized = normalize_algorithm(flat)

    assert normalized[0]["blocks"][0]["type"] == "when_start"
    assert normalized[0]["blocks"][0]["children"][0]["type"] == "set_speed"
    assert normalized[1]["blocks"][0]["children"][0]["type"] == "do_forward"
    assert validate_algorithm(flat).valid


def test_validation_reports_semantic_and_parameter_errors() -> None:
    invalid = [
        {
            "blocks": [
                {"type": "do_forward", "params": {}, "children": []},
                {
                    "type": "when_always",
                    "params": {},
                    "children": [
                        {"type": "set_speed", "params": {"value": 99}, "children": []},
                        {"type": "made_up", "params": {}, "children": []},
                    ],
                },
            ]
        }
    ]

    result = validate_algorithm(invalid)

    assert not result.valid
    assert any("top-level" in error for error in result.errors)
    assert any("children of when_start" in error for error in result.errors)
    assert any("unknown" in error for error in result.errors)


def test_configuration_must_be_directly_under_when_start() -> None:
    invalid = [
        {
            "blocks": [
                {
                    "type": "when_start",
                    "params": {},
                    "children": [
                        {
                            "type": "if_see",
                            "params": {"target": "enemy"},
                            "children": [
                                {"type": "set_speed", "params": {"value": 4.0}, "children": []}
                            ],
                        }
                    ],
                }
            ]
        }
    ]

    result = validate_algorithm(invalid)

    assert not result.valid
    assert any("direct children of when_start" in error for error in result.errors)


def test_ring_placements_are_symmetric_and_legal() -> None:
    placements = generate_ring_placements(5, 300.0, "inward")
    clean, errors, warnings = validate_placements(placements, "farp1")

    assert len(clean) == 5
    assert errors == []
    assert warnings == []
    for placement in clean:
        assert math.hypot(placement["x"] - 1920.0, placement["y"] - 1080.0) == pytest.approx(300.0, abs=0.001)
