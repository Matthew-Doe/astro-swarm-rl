# Training telemetry and bottleneck options

Each PPO update now writes timing fields to `metrics.jsonl`:

- `rollout_seconds`, `inference_seconds`, and `simulation_seconds`
- `update_seconds` (including a device synchronization on CUDA/ROCm)
- outcome counts, clean-goal count, detected-terminal rate, and episode lengths
- attacker-centric shaping, detection, and terminal reward means
- attacker/interceptor reward 5th, 50th, and 95th percentiles

The C++ simulator exposes the three reward components as zero-copy arrays. This keeps logging out of the per-step filesystem path; the only disk write is one JSON record per PPO update.

## Three performance options

1. **Move rollout state and simulation onto the GPU.** Port the hot simulator kernel to HIP or a Torch C++/HIP extension and keep observations/actions on device. This removes the repeated CPU-to-GPU observation copies, which is the highest-potential fix. It is the largest implementation effort and should be benchmarked against the current deterministic CPU reference.

2. **Overlap simulation and learning.** Use two pinned host rollout buffers and a producer/consumer pair: while the GPU updates batch A, OpenMP generates batch B and an asynchronous H2D copy stages the next batch. This addresses the current serialized rollout/inference/update pipeline without changing simulator semantics.

3. **Reduce learner bandwidth and launch overhead.** Keep the C++ simulator, but use packed FP16 observation/action buffers, fused or compiled MLP inference, and larger contiguous minibatches. The current actor networks are small, so reducing transfers and Python tensor-construction overhead is more promising than making the MLP wider. This is the lowest-risk option, with the tradeoff that numerical parity must be checked.

The new phase timings should be collected for a full update on the next run before choosing among these. If `inference_seconds` dominates, prioritize options 1 or 3; if `simulation_seconds` dominates, tune OpenMP/cache behavior or pursue a GPU simulator; if both are small relative to `update_seconds`, optimize the PPO learner and batch layout.
