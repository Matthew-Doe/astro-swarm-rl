# AstroSwarm RL

This directory is a standalone C++/Python reinforcement-learning project. It does not import or call the MCP server. The C++20/OpenMP simulator runs thousands of matches in parallel; PyTorch trains parameter-sharing interceptor and attacker policies on the GPU; exported JSON policies run in a dedicated Godot arena.

## Set up on Arch Linux

```bash
./rl/scripts/bootstrap-arch.sh
```

The script uses `pkexec` for a graphical password prompt when system packages are missing. It installs the native Arch ROCm PyTorch build, creates `rl/.venv` with access to system packages, builds the C++ extension, and performs a real GPU forward/backward check.

## Run

```bash
# CPU simulator throughput
rl/.venv/bin/astroswarm-rl benchmark --envs 32768 --threads 8

# Alternating PPO self-play
rl/.venv/bin/astroswarm-rl train --run-dir rl/runs/first --updates 100

# This machine's 1 TB HDD: local metrics, HDD checkpoints, automatic mounting
./rl/scripts/train-on-hdd.sh first --updates 100

# Continue from a checkpoint with its original simulator/PPO settings
rl/.venv/bin/astroswarm-rl resume rl/runs/first/checkpoints/final.pt \
  --run-dir rl/runs/first --updates 100

# Evaluate, inspect the historical-opponent league, and record trajectories
rl/.venv/bin/astroswarm-rl evaluate rl/runs/first/checkpoints/final.pt
rl/.venv/bin/astroswarm-rl league --run-dir rl/runs/first
rl/.venv/bin/astroswarm-rl record rl/runs/first/checkpoints/final.pt \
  --output rl/runs/first/evaluation.npz

# Export deterministic actors for Godot
rl/.venv/bin/astroswarm-rl export rl/runs/first/checkpoints/final.pt \
  --output rl/runs/first/policies

# Wait for an HDD-backed run to finish, export it, and open the spectator arena
./rl/scripts/watch-run.sh first --seed=9001
```

## Visualize in the separate Godot arena

```bash
godot --path . simulations/rl/AIArena.tscn -- \
  --interceptor-policy=rl/runs/first/policies/interceptor.policy.json \
  --attacker-policy=rl/runs/first/policies/attacker.policy.json \
  --seed=9001
```

Pass `--headless-match` after `--` to print a machine-readable result and exit at the end of one match. Without it, the arena automatically starts a new match three seconds after each result; `SPACE` pauses and `R` resets. `--defenders=1` through `6` changes team size, and `--profile=farp_parity` overrides the profile stored in the policy artifact. The arena duplicates the simulator's observations and policy inference for visualization; training never starts Godot.

The default `balanced_self_play` profile randomizes ship capabilities and uses one to six parameter-sharing interceptors, with five used in 70% of matches. Its sensor FOV is 35–75° (narrower than the original baseline), and defenders are kept outside the planet surface with a small anti-camping penalty. The `farp_parity` profile fixes the original five-ship house preset and 50° FOV for targeted comparisons against the game logic. Optional ship-to-ship collision handling is disabled by default for maximum training throughput; planet exclusion remains enabled.

On this workstation, `train-on-hdd.sh` targets the Btrfs HDD with UUID `929b97d0-d9e7-4b06-a30e-360d09725f7a`. Checkpoints go under `astro-swarm-checkpoints/RUN_NAME`; metrics and league manifests remain in `rl/runs/RUN_NAME` on the NVMe drive. If necessary, `udisksctl` mounts the drive through the desktop authorization prompt.

For telemetry interpretation and the three bottleneck-remediation paths, see [`docs/performance.md`](docs/performance.md).

The no-communication interceptor research assumptions, ablations, and metrics are documented in [`docs/no-communication-research.md`](docs/no-communication-research.md).

To compare four interceptor reward changes sequentially for 15 minutes each, use `./rl/scripts/reward-sweep.sh`. It runs detection, capture-value, track-maintenance, and anti-camping variants in separate HDD-backed run directories.
