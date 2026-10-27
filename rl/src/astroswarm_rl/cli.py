from __future__ import annotations

import argparse
import json
import platform
import signal
import sys
import time
import tomllib
from dataclasses import fields
from pathlib import Path

import numpy as np

from .artifacts import export_actor
from .env import MAX_DEFENDERS, SimulatorConfig, VectorEnv
from .league import League
from .model import make_policy, policy_specs, require_torch
from .training import PPOConfig, SelfPlayTrainer, evaluate_policies


def _load_config(path: Path) -> tuple[SimulatorConfig, PPOConfig]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    simulator_keys = {field.name for field in fields(SimulatorConfig)}
    ppo_keys = {field.name for field in fields(PPOConfig)}
    return (
        SimulatorConfig(**{key: value for key, value in data.get("simulator", {}).items() if key in simulator_keys}),
        PPOConfig(**{key: value for key, value in data.get("ppo", {}).items() if key in ppo_keys}),
    )


def command_doctor(_args) -> int:
    report = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        torch = require_torch()
        report["torch"] = torch.__version__
        report["hip"] = getattr(torch.version, "hip", None)
        report["gpu_available"] = torch.cuda.is_available()
        if not torch.cuda.is_available():
            raise RuntimeError("ROCm PyTorch did not expose a GPU")
        report["gpu"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        report["vram_free_gib"] = round(free / 2**30, 2)
        report["vram_total_gib"] = round(total / 2**30, 2)
        x = torch.randn(4096, 128, device="cuda", requires_grad=True)
        layer = torch.nn.Linear(128, 128, device="cuda")
        loss = layer(x).square().mean()
        loss.backward()
        if not all(torch.isfinite(parameter.grad).all() for parameter in layer.parameters()):
            raise RuntimeError("non-finite ROCm gradient")
        report["forward_backward"] = "ok"
        report["warning"] = "less than 8 GiB VRAM is currently free" if free < 8 * 2**30 else None
    except Exception as exc:
        report["error"] = str(exc)
        print(json.dumps(report, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


def command_benchmark(args) -> int:
    config = SimulatorConfig(
        num_envs=args.envs, threads=args.threads, collisions=args.collisions,
        action_repeat=args.action_repeat, seed=args.seed,
    )
    env = VectorEnv(config)
    interceptor, attacker = env.sample_actions(args.seed)
    for _ in range(10):
        env.step(interceptor, attacker)
    started = time.perf_counter()
    before = env.physics_ticks
    for _ in range(args.steps):
        env.step(interceptor, attacker)
        done = env.arrays["terminated"] | env.arrays["truncated"]
        if np.any(done):
            env.reset_done()
    elapsed = time.perf_counter() - started
    ticks = env.physics_ticks - before
    print(json.dumps({
        "environments": args.envs,
        "threads": args.threads,
        "collisions": args.collisions,
        "physics_ticks": ticks,
        "seconds": round(elapsed, 4),
        "million_physics_ticks_per_second": round(ticks / elapsed / 1e6, 3),
    }, indent=2))
    return 0


def command_train(args) -> int:
    if args.resume:
        torch = require_torch()
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        simulator = SimulatorConfig(**payload["simulator"])
        ppo = PPOConfig(**payload["ppo"])
    else:
        simulator, ppo = _load_config(args.config)
    if args.envs is not None:
        simulator.num_envs = args.envs
    trainer = SelfPlayTrainer(
        simulator, ppo, args.run_dir, checkpoint_directory=args.checkpoint_dir,
        device=args.device, seed=args.seed,
    )
    def request_stop(_signum, _frame) -> None:
        trainer.stop_requested = True
        print("stop requested; saving after the current PPO update", flush=True)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    for metrics in trainer.train(args.updates):
        print(json.dumps(metrics, sort_keys=True), flush=True)
    final = trainer.checkpoint_directory / "final.pt"
    trainer.save_checkpoint(final)
    print(f"checkpoint={final}")
    return 0


def _load_models(checkpoint: Path, device: str):
    torch = require_torch()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    specs = policy_specs()
    interceptor = make_policy(specs["interceptor"]).to(device)
    attacker = make_policy(specs["attacker"]).to(device)
    interceptor.load_state_dict(payload["interceptor"])
    attacker.load_state_dict(payload["attacker"])
    interceptor.eval(); attacker.eval()
    return payload, interceptor, attacker


def command_evaluate(args) -> int:
    payload, interceptor, attacker = _load_models(args.checkpoint, args.device)
    simulator = SimulatorConfig(**payload["simulator"])
    simulator.num_envs = min(args.episodes, args.envs)
    simulator.seed = args.seed
    print(json.dumps(evaluate_policies(interceptor, attacker, simulator, args.episodes, args.device), indent=2))
    return 0


def command_export(args) -> int:
    payload, interceptor, attacker = _load_models(args.checkpoint, "cpu")
    specs = policy_specs()
    profile = payload["simulator"]["profile"]
    export_actor(interceptor, specs["interceptor"], args.output / "interceptor.policy.json", profile, simulator=payload["simulator"])
    export_actor(attacker, specs["attacker"], args.output / "attacker.policy.json", profile, simulator=payload["simulator"])
    print(args.output)
    return 0


def command_league(args) -> int:
    league = League(args.run_dir / "league")
    print(json.dumps(league.summary(), indent=2))
    return 0


def command_record(args) -> int:
    torch = require_torch()
    payload, interceptor, attacker = _load_models(args.checkpoint, args.device)
    simulator = SimulatorConfig(**payload["simulator"])
    simulator.num_envs = 1
    simulator.seed = args.seed
    env = VectorEnv(simulator)
    frames: dict[str, list[np.ndarray | int]] = {
        "episode": [], "step": [], "defender_x": [], "defender_y": [], "defender_heading": [],
        "attacker_x": [], "attacker_y": [], "attacker_heading": [],
        "interceptor_action": [], "attacker_action": [],
    }
    outcomes: list[int] = []
    for episode in range(args.episodes):
        policy_step = 0
        while True:
            interceptor_obs, attacker_obs, masks = env.observations
            with torch.no_grad():
                interceptor_action, _ = interceptor.actor.sample(
                    torch.as_tensor(interceptor_obs, device=args.device).reshape(-1, interceptor_obs.shape[-1]),
                    deterministic=True,
                )
                attacker_action, _ = attacker.actor.sample(
                    torch.as_tensor(attacker_obs, device=args.device), deterministic=True,
                )
            ia = interceptor_action.reshape(1, MAX_DEFENDERS, 2).cpu().numpy()
            aa = attacker_action.cpu().numpy()
            ia *= masks[..., None]
            state = env.state
            frames["episode"].append(episode)
            frames["step"].append(policy_step)
            for key in ("defender_x", "defender_y", "defender_heading"):
                frames[key].append(state[key][0].copy())
            for key in ("attacker_x", "attacker_y", "attacker_heading"):
                frames[key].append(float(state[key][0]))
            frames["interceptor_action"].append(ia[0].copy())
            frames["attacker_action"].append(aa[0].copy())
            result = env.step(ia, aa)
            policy_step += 1
            if bool(result["terminated"][0] or result["truncated"][0]):
                outcomes.append(int(result["outcome"][0]))
                env.reset_done()
                break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **{key: np.asarray(values) for key, values in frames.items()})
    metadata = {
        "version": 1, "checkpoint": str(args.checkpoint), "seed": args.seed,
        "episodes": args.episodes, "outcomes": outcomes,
        "outcome_names": {"1": "interceptors_win", "2": "attacker_goal", "3": "timeout"},
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"trajectory": str(args.output), "frames": len(frames["step"]), **metadata}, indent=2))
    return 0


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[2] / "configs" / "balanced.toml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--envs", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astroswarm-rl")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="verify the native simulator and ROCm PyTorch")
    doctor.set_defaults(func=command_doctor)

    benchmark = commands.add_parser("benchmark", help="measure native simulation throughput")
    benchmark.add_argument("--envs", type=int, default=32_768)
    benchmark.add_argument("--threads", type=int, default=8)
    benchmark.add_argument("--steps", type=int, default=250)
    benchmark.add_argument("--action-repeat", type=int, default=4)
    benchmark.add_argument("--seed", type=int, default=1)
    benchmark.add_argument("--collisions", action="store_true")
    benchmark.set_defaults(func=command_benchmark)

    train = commands.add_parser("train", help="run alternating PPO self-play")
    _add_training_arguments(train)
    train.add_argument("--resume", type=Path)
    train.set_defaults(func=command_train)

    resume = commands.add_parser("resume", help="resume an exact checkpoint configuration")
    resume.add_argument("checkpoint", type=Path)
    _add_training_arguments(resume)
    resume.set_defaults(func=lambda args: command_train(argparse.Namespace(**{**vars(args), "resume": args.checkpoint})))

    evaluate = commands.add_parser("evaluate", help="evaluate a checkpoint on held-out seeds")
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--episodes", type=int, default=10_000)
    evaluate.add_argument("--envs", type=int, default=4096)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--seed", type=int, default=9001)
    evaluate.set_defaults(func=command_evaluate)

    export = commands.add_parser("export", help="export both actors for the Godot arena")
    export.add_argument("checkpoint", type=Path)
    export.add_argument("--output", type=Path, required=True)
    export.set_defaults(func=command_export)

    league = commands.add_parser("league", help="show historical opponents and ratings")
    league.add_argument("--run-dir", type=Path, required=True)
    league.set_defaults(func=command_league)

    record = commands.add_parser("record", help="record deterministic evaluation trajectories")
    record.add_argument("checkpoint", type=Path)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--episodes", type=int, default=10)
    record.add_argument("--device", default="cuda")
    record.add_argument("--seed", type=int, default=9001)
    record.set_defaults(func=command_record)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
