from __future__ import annotations

from pathlib import Path

import pytest

from astroswarm_mcp.algorithm import EXAMPLE_ALGORITHM
from astroswarm_mcp.storage import AlgorithmStore


def test_save_list_and_load(tmp_path: Path) -> None:
    store = AlgorithmStore(tmp_path / "algorithms")

    saved = store.save("Chasing v1", EXAMPLE_ALGORITHM, "baseline")
    listed = store.list()
    loaded = store.load("chasing-v1")

    assert saved["slug"] == "chasing-v1"
    assert listed == [
        {
            "slug": "chasing-v1",
            "name": "Chasing v1",
            "description": "baseline",
            "block_count": 11,
            "has_placements": False,
        }
    ]
    assert loaded["algorithm"] == saved["algorithm"]


def test_save_requires_explicit_overwrite(tmp_path: Path) -> None:
    store = AlgorithmStore(tmp_path)
    store.save("candidate", EXAMPLE_ALGORITHM)

    with pytest.raises(ValueError, match="overwrite=true"):
        store.save("candidate", EXAMPLE_ALGORITHM)

    store.save("candidate", EXAMPLE_ALGORITHM, description="new", overwrite=True)
    assert store.load("candidate")["description"] == "new"
