from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random


@dataclass(slots=True)
class LeagueEntry:
    role: str
    update: int
    checkpoint: str
    rating: float = 1000.0


class League:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.manifest = directory / "league.json"
        self.entries: list[LeagueEntry] = []
        if self.manifest.is_file():
            data = json.loads(self.manifest.read_text(encoding="utf-8"))
            self.entries = [LeagueEntry(**entry) for entry in data.get("entries", [])]

    def add(self, entry: LeagueEntry) -> None:
        self.entries = [
            item for item in self.entries
            if (item.role, item.update, item.checkpoint) != (entry.role, entry.update, entry.checkpoint)
        ]
        self.entries.append(entry)
        self.save()

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest.write_text(
            json.dumps({"version": 1, "entries": [asdict(item) for item in self.entries]}, indent=2) + "\n",
            encoding="utf-8",
        )

    def candidates(self, role: str) -> list[LeagueEntry]:
        return [entry for entry in self.entries if entry.role == role]

    def sample(self, role: str, rng: Random) -> LeagueEntry | None:
        candidates = self.candidates(role)
        return rng.choice(candidates) if candidates else None

    def update_rating(self, entry: LeagueEntry, score: float, opponent_rating: float, k: float = 32.0) -> float:
        expected = 1.0 / (1.0 + 10.0 ** ((opponent_rating - entry.rating) / 400.0))
        old_rating = entry.rating
        entry.rating += k * (score - expected)
        self.save()
        return old_rating

    def summary(self) -> dict:
        by_role = {}
        for role in ("attacker", "interceptor"):
            candidates = self.candidates(role)
            by_role[role] = {
                "snapshots": len(candidates),
                "best_rating": max((entry.rating for entry in candidates), default=None),
                "latest_update": max((entry.update for entry in candidates), default=None),
            }
        return {"version": 1, "roles": by_role, "entries": [asdict(item) for item in self.entries]}
