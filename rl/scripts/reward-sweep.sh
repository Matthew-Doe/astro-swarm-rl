#!/usr/bin/env bash
set -u

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
configs=(
    reward_detection
    reward_capture
    reward_tracking
    reward_anticamp
)

for index in "${!configs[@]}"; do
    name="${configs[index]}"
    run_name="${name}-20260905"
    echo "[$(date -Is)] starting ${run_name}"
    timeout --signal=TERM --kill-after=120s 15m \
        "${project_dir}/rl/scripts/train-on-hdd.sh" "${run_name}" \
        --config "${project_dir}/rl/configs/${name}.toml" \
        --updates 100000 --device cuda --seed "$((20260905 + index + 1))"
    exit_code=$?
    if [[ ${exit_code} -ne 0 && ${exit_code} -ne 124 ]]; then
        echo "[$(date -Is)] ${run_name} exited with ${exit_code}; continuing sweep" >&2
    else
        echo "[$(date -Is)] ${run_name} finished (timeout status ${exit_code})"
    fi
done

echo "[$(date -Is)] reward sweep complete"
