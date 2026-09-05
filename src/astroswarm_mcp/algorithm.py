from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from typing import Any

JsonObject = dict[str, Any]

BLOCKS: dict[str, JsonObject] = {
    "set_speed": {"category": "config", "label": "Set speed", "value": {"min": 0.5, "max": 10.0, "default": 3.75, "unit": "m/s"}},
    "set_turn": {"category": "config", "label": "Set turn rate", "value": {"min": 15.0, "max": 360.0, "default": 120.0, "unit": "deg/s"}},
    "set_view": {"category": "config", "label": "Set vision range", "value": {"min": 0.5, "max": 12.5, "default": 3.75, "unit": "m"}},
    "set_fov": {"category": "config", "label": "Set field of view", "value": {"min": 20.0, "max": 180.0, "default": 90.0, "unit": "degrees"}},
    "when_start": {"category": "event", "label": "On start"},
    "when_always": {"category": "event", "label": "Always"},
    "when_sees": {"category": "event", "label": "When I see anyone"},
    "when_sees_enemy": {"category": "event", "label": "When I see an enemy"},
    "when_sees_ally": {"category": "event", "label": "When I see an ally"},
    "when_alone": {"category": "event", "label": "When I see nobody"},
    "when_near_wall": {"category": "event", "label": "When I touch a wall"},
    "when_sees_wall": {"category": "event", "label": "When I see a wall"},
    "if_see": {"category": "condition", "label": "If I see target", "target": ["anyone", "enemy", "ally", "object", "wall"]},
    "if_within": {"category": "condition", "label": "If target is within range", "value": {"min": 0.5, "max": 12.5, "default": 3.0, "unit": "m"}},
    "if_beyond": {"category": "condition", "label": "If target is beyond range", "value": {"min": 0.5, "max": 12.5, "default": 3.0, "unit": "m"}},
    "else": {"category": "condition", "label": "Else"},
    "do_forward": {"category": "action", "label": "Move forward"},
    "do_backward": {"category": "action", "label": "Move backward"},
    "do_stop": {"category": "action", "label": "Stop"},
    "do_random_walk": {"category": "action", "label": "Random walk"},
    "do_turn_left": {"category": "action", "label": "Turn left continuously", "value": {"min": 15.0, "max": 360.0, "default": 90.0, "unit": "deg/s"}},
    "do_turn_right": {"category": "action", "label": "Turn right continuously", "value": {"min": 15.0, "max": 360.0, "default": 90.0, "unit": "deg/s"}},
    "do_turn_left_by": {"category": "action", "label": "Turn left by angle", "value": {"min": 1.0, "max": 360.0, "default": 180.0, "unit": "degrees"}},
    "do_turn_right_by": {"category": "action", "label": "Turn right by angle", "value": {"min": 1.0, "max": 360.0, "default": 180.0, "unit": "degrees"}},
    "do_face": {"category": "action", "label": "Face the target"},
    "do_flee": {"category": "action", "label": "Flee the target"},
}

LEVELS: dict[str, JsonObject] = {
    "farp1": {
        "name": "Place your defenders",
        "objective": "Capture an evader before it reaches the planet.",
        "placements": "Supply 1-6 defenders between 170 and 465 px from (1920, 1080).",
    },
    "farp2": {
        "name": "Program the ring",
        "objective": "Capture an evader using a fixed five-defender layout.",
        "placements": "Omit placements for the game's deterministic scatter, or supply up to 6 custom placements.",
    },
}

EXAMPLE_ALGORITHM: list[JsonObject] = [
    {
        "x": 40.0,
        "y": 30.0,
        "blocks": [
            {
                "type": "when_start",
                "params": {},
                "children": [
                    {"type": "set_speed", "params": {"value": 5.25}, "children": []},
                    {"type": "set_turn", "params": {"value": 170.0}, "children": []},
                    {"type": "set_view", "params": {"value": 5.5}, "children": []},
                    {"type": "set_fov", "params": {"value": 55.0}, "children": []},
                ],
            }
        ],
    },
    {
        "x": 440.0,
        "y": 30.0,
        "blocks": [
            {
                "type": "when_sees_enemy",
                "params": {},
                "children": [
                    {"type": "do_face", "params": {}, "children": []},
                    {"type": "do_forward", "params": {}, "children": []},
                ],
            }
        ],
    },
    {
        "x": 40.0,
        "y": 310.0,
        "blocks": [
            {
                "type": "when_always",
                "params": {},
                "children": [
                    {"type": "do_forward", "params": {}, "children": []},
                    {"type": "do_turn_left", "params": {"value": 70.0}, "children": []},
                ],
            }
        ],
    },
]


@dataclass(frozen=True)
class ValidationResult:
    algorithm: list[JsonObject]
    errors: list[str]
    warnings: list[str]

    @property
    def valid(self) -> bool:
        return not self.errors

    def as_dict(self) -> JsonObject:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "normalized_algorithm": self.algorithm,
            "block_count": count_blocks(self.algorithm),
            "effective_config": effective_config(self.algorithm),
        }


def algorithm_spec() -> JsonObject:
    return {
        "format": {
            "algorithm": [{"x": 40.0, "y": 30.0, "blocks": ["block"]}],
            "block": {"type": "block_id", "params": {}, "children": ["block"]},
            "rules": [
                "Every top-level stack must begin with a when_* event.",
                "Place set_* configuration blocks under when_start.",
                "Place if_* and else blocks inside an event; else must immediately follow an if_* sibling.",
                "All defenders execute the same algorithm.",
                "Angles in placement.rot are radians; block angles are degrees.",
            ],
        },
        "levels": copy.deepcopy(LEVELS),
        "blocks": copy.deepcopy(BLOCKS),
        "example_algorithm": copy.deepcopy(EXAMPLE_ALGORITHM),
        "example_placements": generate_ring_placements(5, 300.0, "tangent_ccw"),
    }


def normalize_algorithm(data: list[JsonObject]) -> list[JsonObject]:
    if not isinstance(data, list) or not data:
        return []
    if all(isinstance(item, dict) and "blocks" in item for item in data):
        scripts: list[JsonObject] = []
        for index, script in enumerate(data):
            blocks = script.get("blocks", [])
            if not isinstance(blocks, list):
                blocks = []
            scripts.append(
                {
                    "x": _finite_number(script.get("x"), 40.0),
                    "y": _finite_number(script.get("y"), 30.0 + index * 280.0),
                    "blocks": [_normalize_block(block) for block in blocks if isinstance(block, dict)],
                }
            )
        return scripts

    nested: list[JsonObject] = []
    current: JsonObject | None = None
    for item in data:
        if not isinstance(item, dict):
            continue
        block = _normalize_block(item)
        block_type = block["type"]
        if block_type.startswith("when_"):
            current = block
            nested.append(block)
        elif block_type.startswith("do_") and current is not None:
            current["children"].append(block)
        else:
            current = None
            nested.append(block)

    configs = [block for block in nested if block["type"].startswith("set_")]
    heads = [block for block in nested if not block["type"].startswith("set_")]
    if configs:
        heads.insert(0, {"type": "when_start", "params": {}, "children": configs})
    return [
        {
            "x": 40.0 + (index % 2) * 400.0,
            "y": 30.0 + (index // 2) * 280.0,
            "blocks": [block],
        }
        for index, block in enumerate(heads)
    ]


def validate_algorithm(data: list[JsonObject]) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(data, list) or not data:
        return ValidationResult([], ["algorithm must be a non-empty array"], [])
    normalized = normalize_algorithm(data)
    if not normalized:
        return ValidationResult([], ["algorithm contains no usable blocks"], [])

    event_count = 0
    for script_index, script in enumerate(normalized):
        blocks = script["blocks"]
        if not blocks:
            warnings.append(f"scripts[{script_index}] is empty")
            continue
        for block_index, block in enumerate(blocks):
            path = f"scripts[{script_index}].blocks[{block_index}]"
            if not block["type"].startswith("when_"):
                errors.append(f"{path}: top-level block must be a when_* event")
            else:
                event_count += 1
            _validate_block(block, path, errors, warnings, event_ancestor=False, parent_type=None)
    if event_count == 0:
        errors.append("algorithm must contain at least one when_* event")
    return ValidationResult(normalized, errors, warnings)


def validate_placements(placements: list[JsonObject] | None, level_id: str) -> tuple[list[JsonObject], list[str], list[str]]:
    if placements is None:
        if level_id == "farp1":
            return [], ["farp1 requires 1-6 placements"], []
        return [], [], []
    errors: list[str] = []
    warnings: list[str] = []
    clean: list[JsonObject] = []
    if not isinstance(placements, list):
        return [], ["placements must be an array"], []
    if len(placements) > 6:
        errors.append("at most 6 defenders are allowed")
    if level_id == "farp1" and not placements:
        errors.append("farp1 requires at least one defender")
    for index, placement in enumerate(placements[:6]):
        if not isinstance(placement, dict):
            errors.append(f"placements[{index}] must be an object")
            continue
        values: dict[str, float] = {}
        for key in ("x", "y", "rot"):
            value = placement.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                errors.append(f"placements[{index}].{key} must be a finite number")
                values[key] = 0.0
            else:
                values[key] = float(value)
        if not (0.0 <= values["x"] <= 3840.0 and 0.0 <= values["y"] <= 2160.0):
            errors.append(f"placements[{index}] lies outside the 3840x2160 arena")
        radius = math.hypot(values["x"] - 1920.0, values["y"] - 1080.0)
        legal_max = 465.0 if level_id == "farp1" else 250.0
        if not 170.0 <= radius <= legal_max:
            warnings.append(
                f"placements[{index}] is {radius:.1f}px from the planet; the {level_id} UI permits 170-{legal_max:.0f}px"
            )
        clean.append(values)
    return clean, errors, warnings


def generate_ring_placements(count: int, radius: float, facing: str) -> list[JsonObject]:
    if not 1 <= count <= 6:
        raise ValueError("count must be between 1 and 6")
    if not 170.0 <= radius <= 465.0:
        raise ValueError("radius must be between 170 and 465 pixels")
    offsets = {"outward": 0.0, "inward": math.pi, "tangent_cw": math.pi / 2.0, "tangent_ccw": -math.pi / 2.0}
    if facing not in offsets:
        raise ValueError("facing must be inward, outward, tangent_cw, or tangent_ccw")
    result: list[JsonObject] = []
    for index in range(count):
        angle = math.tau * index / count
        result.append(
            {
                "x": round(1920.0 + math.cos(angle) * radius, 4),
                "y": round(1080.0 + math.sin(angle) * radius, 4),
                "rot": round(angle + offsets[facing], 6),
            }
        )
    return result


def count_blocks(algorithm: list[JsonObject]) -> int:
    def count(nodes: list[JsonObject]) -> int:
        return sum(1 + count(node.get("children", [])) for node in nodes)

    return sum(count(script.get("blocks", [])) for script in algorithm)


def effective_config(algorithm: list[JsonObject]) -> JsonObject:
    config: JsonObject = {"speed_m_s": 3.75, "turn_deg_s": round(math.degrees(3.2), 3), "view_m": 7.5, "fov_deg": 70.0}
    mapping = {"set_speed": "speed_m_s", "set_turn": "turn_deg_s", "set_view": "view_m", "set_fov": "fov_deg"}
    for script in algorithm:
        for event in script.get("blocks", []):
            if event.get("type") != "when_start":
                continue
            for block in event.get("children", []):
                key = mapping.get(block.get("type"))
                if key is not None:
                    config[key] = float(block.get("params", {}).get("value", config[key]))
    return config


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    if not slug:
        raise ValueError("name must contain at least one letter or number")
    return slug[:64]


def _normalize_block(block: JsonObject) -> JsonObject:
    children = block.get("children", [])
    if not isinstance(children, list):
        children = []
    params = block.get("params", {})
    if not isinstance(params, dict):
        params = {}
    return {
        "type": str(block.get("type", "")),
        "params": copy.deepcopy(params),
        "children": [_normalize_block(child) for child in children if isinstance(child, dict)],
    }


def _validate_block(
    block: JsonObject,
    path: str,
    errors: list[str],
    warnings: list[str],
    *,
    event_ancestor: bool,
    parent_type: str | None,
) -> None:
    block_type = block["type"]
    definition = BLOCKS.get(block_type)
    if definition is None:
        errors.append(f"{path}: unknown or unavailable FARP block {block_type!r}")
        return
    category = definition["category"]
    now_event = event_ancestor or block_type.startswith("when_")
    if category == "condition" and not event_ancestor:
        errors.append(f"{path}: condition blocks must be nested inside an event")
    if category == "config" and parent_type != "when_start":
        errors.append(f"{path}: configuration blocks must be direct children of when_start")
    if category == "action" and not event_ancestor:
        errors.append(f"{path}: action blocks must be nested inside an event")

    params = block.get("params", {})
    value_spec = definition.get("value")
    if value_spec is not None:
        value = params.get("value", value_spec["default"])
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            errors.append(f"{path}.params.value must be a finite number")
        elif not value_spec["min"] <= float(value) <= value_spec["max"]:
            errors.append(f"{path}.params.value must be between {value_spec['min']} and {value_spec['max']}")
    if block_type == "if_see" and params.get("target", "anyone") not in definition["target"]:
        errors.append(f"{path}.params.target is not a supported target")

    children = block.get("children", [])
    if category in {"action", "config"} and children:
        warnings.append(f"{path}: {category} block children are ignored")
    for index, child in enumerate(children):
        if child.get("type") == "else" and (index == 0 or not children[index - 1].get("type", "").startswith("if_")):
            errors.append(f"{path}.children[{index}]: else must immediately follow an if_* block")
        _validate_block(
            child,
            f"{path}.children[{index}]",
            errors,
            warnings,
            event_ancestor=now_event,
            parent_type=block_type,
        )


def _finite_number(value: Any, default: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return default
