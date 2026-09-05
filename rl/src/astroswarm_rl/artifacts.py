from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .model import PolicySpec, require_torch


def _linear_layers(module) -> list[dict[str, Any]]:
    torch = require_torch()
    layers: list[dict[str, Any]] = []
    for child in module:
        if isinstance(child, torch.nn.Linear):
            layers.append(
                {
                    "activation": "tanh",
                    "input": child.in_features,
                    "output": child.out_features,
                    "weight": child.weight.detach().cpu().float().flatten().tolist(),
                    "bias": child.bias.detach().cpu().float().tolist(),
                }
            )
    return layers


def export_actor(model, spec: PolicySpec, destination: Path, profile: str) -> dict[str, Any]:
    actor = model.actor
    payload: dict[str, Any] = {
        "format": "astroswarm-policy",
        "version": 1,
        "role": spec.role,
        "profile": profile,
        "observation": {
            "obs_dim": spec.obs_dim,
            "self_dim": spec.self_dim,
            "entity_count": spec.entity_count,
            "entity_dim": spec.entity_dim,
            "mask_index": spec.mask_index,
        },
        "action": {"names": ["throttle", "turn"], "minimum": -1.0, "maximum": 1.0},
        "network": {
            "self": _linear_layers(actor.self_encoder),
            "entity": _linear_layers(actor.entity_encoder),
            "trunk": _linear_layers(actor.trunk),
            "output": {
                "activation": "tanh",
                "input": actor.mean.in_features,
                "output": actor.mean.out_features,
                "weight": actor.mean.weight.detach().cpu().float().flatten().tolist(),
                "bias": actor.mean.bias.detach().cpu().float().tolist(),
            },
        },
        "normalization": {"kind": "fixed", "coordinates": "planet-centered-and-scaled"},
    }
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    payload["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_checkpoint(path: Path, device: str = "cpu"):
    torch = require_torch()
    return torch.load(path, map_location=device, weights_only=False)
