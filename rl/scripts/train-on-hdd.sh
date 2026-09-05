#!/usr/bin/env bash
set -euo pipefail

hdd_uuid="929b97d0-d9e7-4b06-a30e-360d09725f7a"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if (( $# < 1 )); then
    echo "usage: $0 RUN_NAME [training options]" >&2
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
mkdir -p "${checkpoint_dir}"
exec "${project_dir}/rl/.venv/bin/astroswarm-rl" train \
    --run-dir "${project_dir}/rl/runs/${run_name}" \
    --checkpoint-dir "${checkpoint_dir}" \
    "$@"
