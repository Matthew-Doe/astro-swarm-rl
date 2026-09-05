from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .algorithm import JsonObject, slugify, validate_algorithm, validate_placements


class AlgorithmStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()

    def list(self) -> list[JsonObject]:
        if not self.directory.exists():
            return []
        entries: list[JsonObject] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                payload = self._read(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            entries.append(
                {
                    "slug": path.stem,
                    "name": payload.get("name", path.stem),
                    "description": payload.get("description", ""),
                    "block_count": payload.get("block_count", 0),
                    "has_placements": bool(payload.get("placements")),
                }
            )
        return entries

    def load(self, name: str) -> JsonObject:
        path = self._path(name)
        if not path.is_file():
            raise ValueError(f"saved algorithm {name!r} does not exist")
        return self._read(path)

    def save(
        self,
        name: str,
        algorithm: list[JsonObject],
        description: str = "",
        placements: list[JsonObject] | None = None,
        *,
        overwrite: bool = False,
    ) -> JsonObject:
        validation = validate_algorithm(algorithm)
        if not validation.valid:
            raise ValueError("invalid algorithm: " + "; ".join(validation.errors))
        clean_placements, placement_errors, placement_warnings = validate_placements(placements, "farp2")
        if placement_errors:
            raise ValueError("invalid placements: " + "; ".join(placement_errors))
        path = self._path(name)
        if path.exists() and not overwrite:
            raise ValueError(f"saved algorithm {path.stem!r} already exists; set overwrite=true to replace it")
        self.directory.mkdir(parents=True, exist_ok=True)
        validation_details = validation.as_dict()
        payload: JsonObject = {
            "name": name.strip(),
            "slug": path.stem,
            "description": description.strip(),
            "algorithm": validation.algorithm,
            "placements": clean_placements,
            "block_count": validation_details["block_count"],
            "effective_config": validation_details["effective_config"],
            "warnings": validation.warnings + placement_warnings,
        }
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary_name, path)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return payload

    def _path(self, name: str) -> Path:
        return self.directory / f"{slugify(name)}.json"

    @staticmethod
    def _read(path: Path) -> JsonObject:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path.name} must contain a JSON object")
        return payload
