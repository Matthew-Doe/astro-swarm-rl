from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from pydantic import Field

from .algorithm import (
    JsonObject,
    algorithm_spec,
    generate_ring_placements as make_ring,
    validate_algorithm as check_algorithm,
    validate_placements,
)
from .benchmark import BenchmarkRunner
from .storage import AlgorithmStore


def _project_root() -> Path:
    override = os.environ.get("ASTROSWARM_PROJECT_ROOT")
    return Path(override).expanduser().resolve() if override else Path(__file__).resolve().parents[2]


PROJECT_ROOT = _project_root()
store = AlgorithmStore(PROJECT_ROOT / "algorithms")
runner = BenchmarkRunner(PROJECT_ROOT)

mcp = MCPServer(
    "AstroSwarm Algorithm Lab",
    version="0.1.0",
    instructions=(
        "Develop AstroSwarm defender algorithms with the block specification, validate each candidate, "
        "save useful versions, and benchmark them with fixed seeds. Compare candidates with identical level, "
        "placements, seed, trials, and collision settings. Start with small trial counts, then confirm with more trials."
    ),
)


@mcp.tool()
async def get_algorithm_spec() -> JsonObject:
    """Return the complete FARP algorithm format, available blocks, constraints, levels, and a working example."""
    return algorithm_spec()


@mcp.tool()
async def validate_algorithm(algorithm: list[dict[str, Any]]) -> JsonObject:
    """Validate and normalize an algorithm without running Godot. Fix all errors before benchmarking."""
    return check_algorithm(algorithm).as_dict()


@mcp.tool()
async def generate_ring_placements(
    count: Annotated[int, Field(ge=1, le=6, description="Number of defenders")] = 5,
    radius: Annotated[
        float,
        Field(ge=170.0, le=465.0, description="Distance from planet center in pixels"),
    ] = 300.0,
    facing: Literal["inward", "outward", "tangent_cw", "tangent_ccw"] = "tangent_ccw",
) -> JsonObject:
    """Generate a symmetric legal defender layout for repeatable Level 1 experiments."""
    return {"placements": make_ring(count, radius, facing)}


@mcp.tool()
async def save_algorithm(
    name: Annotated[str, Field(min_length=1, max_length=80)],
    algorithm: list[dict[str, Any]],
    description: Annotated[str, Field(max_length=1000)] = "",
    placements: list[dict[str, Any]] | None = None,
    overwrite: bool = False,
) -> JsonObject:
    """Save a valid named candidate under the repository's algorithms directory for later iterations."""
    return store.save(name, algorithm, description, placements, overwrite=overwrite)


@mcp.tool()
async def list_algorithms() -> JsonObject:
    """List saved algorithm candidates and whether each includes a defender layout."""
    entries = store.list()
    return {"algorithms": entries, "count": len(entries)}


@mcp.tool()
async def load_algorithm(name: str) -> JsonObject:
    """Load a saved algorithm candidate by its name or slug."""
    return store.load(name)


@mcp.tool()
async def benchmark_algorithm(
    algorithm: list[dict[str, Any]],
    level_id: Literal["farp1", "farp2"] = "farp2",
    placements: list[dict[str, Any]] | None = None,
    trials: Annotated[int, Field(ge=1, le=200)] = 10,
    seed: int = 987654321,
    match_seconds: Annotated[float, Field(ge=1.0, le=240.0)] = 240.0,
    collisions: bool = False,
    sweep_count: Annotated[
        int,
        Field(
            ge=0,
            le=100,
            description="Also test ring sizes 1 through this count; 0 disables the sweep",
        ),
    ] = 0,
    sweep_trials: Annotated[int, Field(ge=1, le=50)] = 3,
) -> JsonObject:
    """Run a deterministic, faster-than-real-time Godot benchmark and return compact performance metrics."""
    return await _benchmark(
        algorithm,
        level_id,
        placements,
        trials,
        seed,
        match_seconds,
        collisions,
        sweep_count,
        sweep_trials,
    )


@mcp.tool()
async def benchmark_saved_algorithm(
    name: str,
    level_id: Literal["farp1", "farp2"] = "farp2",
    trials: Annotated[int, Field(ge=1, le=200)] = 10,
    seed: int = 987654321,
    match_seconds: Annotated[float, Field(ge=1.0, le=240.0)] = 240.0,
    collisions: bool = False,
    sweep_count: Annotated[int, Field(ge=0, le=100)] = 0,
    sweep_trials: Annotated[int, Field(ge=1, le=50)] = 3,
) -> JsonObject:
    """Benchmark a saved candidate, reusing its saved placements when present."""
    payload = store.load(name)
    return await _benchmark(
        payload["algorithm"],
        level_id,
        payload.get("placements") or None,
        trials,
        seed,
        match_seconds,
        collisions,
        sweep_count,
        sweep_trials,
    )


async def _benchmark(
    algorithm: list[dict[str, Any]],
    level_id: str,
    placements: list[dict[str, Any]] | None,
    trials: int,
    seed: int,
    match_seconds: float,
    collisions: bool,
    sweep_count: int,
    sweep_trials: int,
) -> JsonObject:
    validation = check_algorithm(algorithm)
    if not validation.valid:
        raise ValueError("invalid algorithm: " + "; ".join(validation.errors))
    clean_placements, errors, warnings = validate_placements(placements, level_id)
    if errors:
        raise ValueError("invalid placements: " + "; ".join(errors))
    result = await runner.run(
        validation.algorithm,
        level_id=level_id,
        placements=clean_placements,
        trials=trials,
        seed=seed,
        match_seconds=match_seconds,
        collisions=collisions,
        sweep_count=sweep_count,
        sweep_trials=sweep_trials,
    )
    result["warnings"] = validation.warnings + warnings
    return result


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
