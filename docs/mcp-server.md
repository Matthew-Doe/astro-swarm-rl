# AstroSwarm MCP server

This stdio MCP server gives an agent a closed loop for building defender algorithms:

1. Call `get_algorithm_spec` to learn the JSON format and legal blocks.
2. Draft a candidate and call `validate_algorithm`.
3. Call `benchmark_algorithm` with a small trial count and a fixed seed.
4. Save useful candidates with `save_algorithm` and compare them using identical benchmark settings.
5. Increase the trial count or enable a defender-count sweep for final confirmation.

## Run

From the repository root:

```bash
uv sync
uv run astroswarm-mcp
```

An MCP client configuration can launch it with:

```json
{
  "command": "uv",
  "args": ["--directory", "/home/matthewd/devel/astro-swarm", "run", "astroswarm-mcp"]
}
```

The server uses `godot` from `PATH`. Set `GODOT_BIN` to override the executable and
`ASTROSWARM_PROJECT_ROOT` when launching the installed package away from this checkout.

Saved candidates live in `algorithms/*.json`. Benchmark inputs and outputs use private temporary
directories and are deleted after each call. The server captures Godot's output so it never corrupts
the MCP stdio protocol.
