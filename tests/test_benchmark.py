from __future__ import annotations

from pathlib import Path

import pytest

from astroswarm_mcp.algorithm import EXAMPLE_ALGORITHM
from astroswarm_mcp.benchmark import BenchmarkRunner, summarize_result


def test_result_summary_drops_replay_frames() -> None:
    payload = {
        "level_id": "farp2",
        "results": {
            "outcomes": ["win", "lose", "timeout"],
            "success_rate": 33.3,
            "detection_times": [2.0, -1.0, 4.0],
            "capture_times": [3.0, -1.0, -1.0],
            "goal_times": [-1.0, 8.25, -1.0],
            "sweep": [],
        },
        "replays": {
            "defenders": 5,
            "runs": [
                {"trial": 0, "outcome": "win", "detection_time": 2.0, "capture_time": 3.0, "goal_time": -1.0, "frames": [[1, 2, 3]]}
            ],
        },
    }

    result = summarize_result(payload, EXAMPLE_ALGORITHM, [], 42, False, 0.2)

    assert result["summary"]["wins"] == 1
    assert result["summary"]["detection_rate"] == 66.7
    assert result["summary"]["mean_detection_time"] == 3.0
    assert "frames" not in result["trials"][0]


@pytest.mark.integration
async def test_godot_benchmark_smoke(project_root: Path = Path(__file__).resolve().parents[1]) -> None:
    runner = BenchmarkRunner(project_root)

    result = await runner.run(
        EXAMPLE_ALGORITHM,
        level_id="farp2",
        placements=[],
        trials=1,
        seed=1234,
        match_seconds=20.0,
        collisions=False,
        sweep_count=0,
        sweep_trials=1,
    )

    assert result["level_id"] == "farp2"
    assert result["summary"]["trials"] == 1
    assert result["duration_seconds"] < 30.0
