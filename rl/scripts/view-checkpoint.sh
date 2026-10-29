#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if (( $# < 1 )); then
    echo "usage: $0 CHECKPOINT.pt_OR_RUN_DIRECTORY [--headless] [--seed=N] [--defenders=N] [--headless-match]" >&2
    exit 2
fi
source_path="$1"
shift
checkpoint="$("${project_dir}/rl/.venv/bin/python" - "$source_path" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
if p.is_dir():
    folder = p / 'checkpoints' if (p / 'checkpoints').is_dir() else p
    final = folder / 'final.pt'
    snapshots = sorted(folder.glob('update-*.pt'))
    p = final if final.is_file() else (snapshots[-1] if snapshots else folder / 'final.pt')
if not p.is_file():
    sys.exit(f'No checkpoint found: {p}')
print(p.resolve())
PY
)"
policy_dir="$(mktemp -d "${TMPDIR:-/tmp}/astroswarm-view.XXXXXX")"
trap 'rm -rf -- "$policy_dir"' EXIT
"${project_dir}/rl/.venv/bin/astroswarm-rl" export "$checkpoint" --output "$policy_dir"
if [[ -f "${policy_dir}/attacker-controller.json" ]]; then
    echo "This checkpoint uses the Python planning attacker, which the Godot viewer does not support." >&2
    exit 1
fi
godot_options=()
arena_options=()
for argument in "$@"; do
    if [[ "$argument" == "--headless" ]]; then
        godot_options+=(--headless)
    else
        arena_options+=("$argument")
    fi
done
echo "Viewing checkpoint: ${checkpoint}"
godot "${godot_options[@]}" --path "$project_dir" "${project_dir}/simulations/rl/AIArena.tscn" -- \
    "--interceptor-policy=${policy_dir}/interceptor.policy.json" \
    "--attacker-policy=${policy_dir}/attacker.policy.json" "${arena_options[@]}"
