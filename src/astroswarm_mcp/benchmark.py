from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from statistics import fmean
from typing import Any

from .algorithm import JsonObject, count_blocks, effective_config


class BenchmarkError(RuntimeError):
    """Raised when the Godot benchmark cannot complete."""


class BenchmarkRunner:
    def __init__(self, project_root: Path, godot_binary: str | None = None) -> None:
        self.project_root = project_root.resolve()
        self.godot_binary = godot_binary or os.environ.get("GODOT_BIN") or shutil.which("godot") or "godot"

    async def run(
        self,
        algorithm: list[JsonObject],
        *,
        level_id: str,
        placements: list[JsonObject],
        trials: int,
        seed: int,
        match_seconds: float,
        collisions: bool,
        sweep_count: int,
        sweep_trials: int,
        timeout_seconds: float = 180.0,
    ) -> JsonObject:
        if not (self.project_root / "project.godot").is_file():
            raise BenchmarkError(f"Godot project not found at {self.project_root}")
        if shutil.which(self.godot_binary) is None and not Path(self.godot_binary).is_file():
            raise BenchmarkError(f"Godot executable not found: {self.godot_binary}")

        with tempfile.TemporaryDirectory(prefix="astroswarm-mcp-") as temporary:
            work = Path(temporary)
            algorithm_path = work / "algorithm.json"
            placements_path = work / "placements.json"
            output_path = work / "result.json"
            algorithm_path.write_text(json.dumps({"algorithm": algorithm}), encoding="utf-8")
            placements_path.write_text(json.dumps({"placements": placements}), encoding="utf-8")
            command = [
                self.godot_binary,
                "--headless",
                "--quiet",
                "--fixed-fps",
                "60",
                "--path",
                str(self.project_root),
                "--",
                "--bench",
                f"--level-id={level_id}",
                f"--algorithm={algorithm_path}",
                f"--placements={placements_path}",
                f"--out={output_path}",
                f"--trials={trials}",
                f"--trial-count={trials}",
                "--n-start=1",
                f"--n-count={sweep_count}",
                f"--sweep-trials={sweep_trials}",
                f"--seed={seed}",
                f"--match-seconds={match_seconds}",
                "--goal-tail-seconds=0",
                f"--collisions={int(collisions)}",
                "--record-replays=0",
            ]
            environment = os.environ.copy()
            environment.update(
                {
                    "XDG_CONFIG_HOME": str(work / "xdg" / "config"),
                    "XDG_CACHE_HOME": str(work / "xdg" / "cache"),
                    "XDG_DATA_HOME": str(work / "xdg" / "data"),
                }
            )
            started = time.monotonic()
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=self.project_root,
                    env=environment,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
            except TimeoutError as exc:
                process.kill()
                await process.communicate()
                raise BenchmarkError(f"benchmark exceeded {timeout_seconds:.0f}s timeout") from exc
            duration = time.monotonic() - started
            stdout = stdout_bytes.decode(errors="replace")
            stderr = stderr_bytes.decode(errors="replace")
            if process.returncode != 0:
                details = (stderr or stdout)[-4000:].strip()
                raise BenchmarkError(f"Godot benchmark failed with exit code {process.returncode}: {details}")
            if not output_path.is_file():
                details = (stderr or stdout)[-4000:].strip()
                raise BenchmarkError(f"Godot produced no result file: {details}")
            try:
                payload: Any = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise BenchmarkError(f"could not read Godot benchmark result: {exc}") from exc
        if not isinstance(payload, dict):
            raise BenchmarkError("Godot benchmark result was not a JSON object")
        return summarize_result(payload, algorithm, placements, seed, collisions, duration)


def summarize_result(
    payload: JsonObject,
    algorithm: list[JsonObject],
    placements: list[JsonObject],
    seed: int,
    collisions: bool,
    duration_seconds: float,
) -> JsonObject:
    results = payload.get("results", {})
    outcomes = list(results.get("outcomes", []))
    runs = payload.get("replays", {}).get("runs", [])
    per_trial = [
        {
            "trial": run.get("trial"),
            "outcome": run.get("outcome"),
            "detection_time": run.get("detection_time"),
            "capture_time": run.get("capture_time"),
            "goal_time": run.get("goal_time"),
        }
        for run in runs
    ]
    detection_times = _positive_times(results.get("detection_times", []))
    capture_times = _positive_times(results.get("capture_times", []))
    goal_times = _positive_times(results.get("goal_times", []))
    trial_count = len(outcomes)
    return {
        "level_id": payload.get("level_id"),
        "seed": seed,
        "collisions": collisions,
        "algorithm": {"block_count": count_blocks(algorithm), "effective_config": effective_config(algorithm)},
        "placements": len(placements) if placements else payload.get("replays", {}).get("defenders", 0),
        "summary": {
            "trials": trial_count,
            "wins": outcomes.count("win"),
            "losses": outcomes.count("lose"),
            "timeouts": outcomes.count("timeout"),
            "success_rate": results.get("success_rate", 0.0),
            "detection_rate": _rate(detection_times, trial_count),
            "capture_rate": _rate(capture_times, trial_count),
            "mean_detection_time": _mean(detection_times),
            "mean_capture_time": _mean(capture_times),
            "mean_goal_time": _mean(goal_times),
        },
        "trials": per_trial,
        "sweep": results.get("sweep", []),
        "duration_seconds": round(duration_seconds, 3),
    }


def _positive_times(values: list[Any]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float)) and float(value) >= 0.0]


def _rate(values: list[float], total: int) -> float:
    return round(100.0 * len(values) / total, 1) if total else 0.0


def _mean(values: list[float]) -> float | None:
    return round(fmean(values), 3) if values else None
