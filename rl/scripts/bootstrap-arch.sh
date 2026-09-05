#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

packages=(base-devel cmake ninja uv python-pytorch-opt-rocm)
missing=()
for package in "${packages[@]}"; do
    pacman -Q "${package}" >/dev/null 2>&1 || missing+=("${package}")
done
if (( ${#missing[@]} )); then
    pkexec pacman -S --needed --noconfirm "${missing[@]}"
fi

uv venv --clear --python /usr/bin/python --system-site-packages "${project_dir}/.venv"
UV_PROJECT_ENVIRONMENT="${project_dir}/.venv" uv sync \
    --project "${project_dir}" --extra train --extra dev
"${project_dir}/.venv/bin/astroswarm-rl" doctor
