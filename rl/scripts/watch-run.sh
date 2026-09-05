#!/usr/bin/env bash
set -euo pipefail

hdd_uuid="929b97d0-d9e7-4b06-a30e-360d09725f7a"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if (( $# < 1 )); then
    echo "usage: $0 RUN_NAME [--seed=N] [--defenders=N] [--profile=PROFILE]" >&2
    exit 2
fi
run_name="$1"
shift
if [[ ! "${run_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_NAME may contain only letters, numbers, dot, underscore, and hyphen" >&2
    exit 2
fi

mount_point="$(findmnt -rn -S "UUID=${hdd_uuid}" -o TARGET || true)"
if [[ -z "${mount_point}" ]]; then
    udisksctl mount -b "/dev/disk/by-uuid/${hdd_uuid}" >/dev/null
    mount_point="$(findmnt -rn -S "UUID=${hdd_uuid}" -o TARGET)"
fi

checkpoint_dir="${mount_point}/astro-swarm-checkpoints/${run_name}"
run_dir="${project_dir}/rl/runs/${run_name}"
mkdir -p "${run_dir}"
checkpoint="${checkpoint_dir}/final.pt"
if [[ ! -f "${checkpoint}" ]]; then
	 echo "Waiting for the training process to finish: ${run_name}"
	 while pgrep -f -- "astroswarm-rl train.*${run_name}" >/dev/null 2>&1; do
	     sleep 15
	 done
	 if [[ ! -f "${checkpoint}" ]]; then
	     checkpoint="$(find "${checkpoint_dir}" -maxdepth 1 -type f -name 'update-*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)"
	 fi
fi

if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
	 echo "No completed checkpoint found in ${checkpoint_dir}" >&2
	 exit 1
fi

policy_dir="${run_dir}/policies"
"${project_dir}/rl/.venv/bin/astroswarm-rl" export "${checkpoint}" --output "${policy_dir}"
exec godot --path "${project_dir}" "${project_dir}/simulations/rl/AIArena.tscn" -- \
    "--interceptor-policy=${policy_dir}/interceptor.policy.json" \
    "--attacker-policy=${policy_dir}/attacker.policy.json" \
    "$@"
