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

## Combined tracking, anti-camping, and sneaky goals

`configs/reward_stealth.toml` combines tracking reward `0.003` with the 10×
anti-camping penalty (`0.01`). Interceptor FOV stays **35–75°**. Attacker goals
pay a terminal reward of **5.0 if never detected during the episode**, or **1.0
if detected at any point**, including the goal tick. The interceptor team
receives the opposite reward, preserving zero-sum training. Other shaping and
reward components are still added separately. `clean_goal_count` tracks these
undetected successes, and `terminal_reward_mean` includes the larger payout.

```bash
./rl/scripts/train-on-hdd.sh reward_stealth --config rl/configs/reward_stealth.toml --updates 100
```

This starts a new run with the combined settings. Resuming an older checkpoint
retains its original simulator/PPO configuration. Configs and checkpoints that
omit `clean_goal_reward` retain the previous clean-goal reward of `1.1`.

Actor exports now include saved simulator settings so the visual arena uses the
training FOV. Older exports without these settings fall back to 35–75° for
`balanced_self_play`; `farp_parity` retains its fixed 50° FOV.

## Defensive-learning reward preset

`configs/reward_defense.toml` retains the 35–75° FOV and +5 never-detected
attacker goal, and adds:

- Capture reward **+3** to defenders; timeout reward **+0.25**.
- Conditional anti-camping: a defender must spend **over two seconds** near the
  planet with low throttle and no visual contact before being penalized.
  Movement, contact, or leaving the camping radius resets that defender's timer.
  The `0.01` team penalty scales with the fraction of active defenders camping
  and the fraction of physics ticks in the action step that qualify.
- Team-contact tracking reward **+0.003** per full action step with at least one
  defender seeing the attacker. Additional observers earn no extra reward.
  Total tracking payments are capped at **+0.3 per episode**.
- Pursuit shaping with strength **0.5**:
  `0.5 * (gamma * Phi(next) - Phi(current))`, where
  `Phi = 1 - clamp(nearest_defender_distance / 1000, 0, 1)`.
  This uses global simulator state during training, without adding actor inputs
  or communication. All terminal potentials, including timeouts, are zero.
  `reward_shaping_gamma` must match PPO's decision-step discount (`0.995`).
  The existing `closing_reward_mean` metric records this term, attacker-centric.

Reward components remain zero-sum. These changes retain team advantages and
shared actors; individual credit assignment would require a separate learner
change. Older configs default to the previous camping/tracking/timeout modes.
Legacy distance-delta closing shaping now correctly charges for retreat as well
as rewarding approach. Finished environments clear all per-step reward metrics.

```bash
./rl/scripts/train-on-hdd.sh reward_defense --config rl/configs/reward_defense.toml --updates 100
```
