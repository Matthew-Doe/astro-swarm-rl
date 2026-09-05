from __future__ import annotations

from mcp import Client

from astroswarm_mcp.server import mcp


async def test_mcp_tool_discovery_and_call() -> None:
    async with Client(mcp, raise_exceptions=True) as client:
        listed = await client.list_tools()
        names = {tool.name for tool in listed.tools}

        assert {
            "get_algorithm_spec",
            "validate_algorithm",
            "generate_ring_placements",
            "save_algorithm",
            "list_algorithms",
            "load_algorithm",
            "benchmark_algorithm",
            "benchmark_saved_algorithm",
        } <= names

        result = await client.call_tool("generate_ring_placements", {"count": 3, "facing": "inward"})
        assert result.is_error is False
        assert len(result.structured_content["placements"]) == 3
