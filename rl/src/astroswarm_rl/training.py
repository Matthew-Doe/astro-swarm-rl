from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .env import (
    ATTACKER_OBS_DIM,
    MAX_DEFENDERS,
    SimulatorConfig,
    VectorEnv,
    orbit_interceptor_actions,
    straight_attacker_actions,
)
from .planner import planning_attacker_actions
from .league import League, LeagueEntry
from .model import make_policy, policy_specs, require_torch


@dataclass(slots=True)
class PPOConfig:
    attacker_controller: str = "learned"
    architecture: str = "mlp"
    spike_steps: int = 8
    rollout_steps: int = 128
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    learning_rate: float = 3.0e-4
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch_size: int = 65_536
    phase_updates: int = 10
    snapshot_interval: int = 10
    bootstrap_updates: int = 4
    historical_opponent_probability: float = 0.25


class SelfPlayTrainer:
    def __init__(
        self,
        simulator: SimulatorConfig,
        ppo: PPOConfig,
        run_directory: Path,
        *,
        checkpoint_directory: Path | None = None,
        device: str = "cuda",
        seed: int = 1,
    ) -> None:
        if simulator.closing_potential_shaping and simulator.closing_reward > 0 and abs(simulator.reward_shaping_gamma - ppo.gamma) > 1.0e-9:
            raise ValueError("reward_shaping_gamma must match PPO gamma for potential shaping")
        if ppo.attacker_controller not in ("learned", "planner"):
            raise ValueError("attacker_controller must be learned or planner")
        self.torch = require_torch()
        self.device = self.torch.device(device)
        self.simulator_config = simulator
        self.ppo = ppo
        self.run_directory = run_directory
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.checkpoint_directory = checkpoint_directory or (run_directory / "checkpoints")
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        self.env = VectorEnv(simulator)
        specs = policy_specs()
        self.interceptor = make_policy(specs["interceptor"], ppo.architecture, ppo.spike_steps).to(self.device)
        self.attacker = make_policy(specs["attacker"], ppo.architecture, ppo.spike_steps).to(self.device)
        self.interceptor_optimizer = self.torch.optim.Adam(
            self.interceptor.parameters(), lr=ppo.learning_rate
        )
        self.attacker_optimizer = self.torch.optim.Adam(
            self.attacker.parameters(), lr=ppo.learning_rate
        )
        self.league = League(run_directory / "league")
        self.current_rating = {"attacker": 1000.0, "interceptor": 1000.0}
        self.stop_requested = False
        self.update_index = 0
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)
        self.metrics_path = run_directory / "metrics.jsonl"
        self.last_rollout_timing = {"rollout_seconds": 0.0, "inference_seconds": 0.0, "simulation_seconds": 0.0}
        self._write_run_metadata()

    def _write_run_metadata(self) -> None:
        metadata = {
            "version": 1,
            "simulator": asdict(self.simulator_config),
            "ppo": asdict(self.ppo),
            "device": str(self.device),
            "seed": self.seed,
            "checkpoint_directory": str(self.checkpoint_directory),
            "torch": self.torch.__version__,
            "hip": getattr(self.torch.version, "hip", None),
        }
        (self.run_directory / "run.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    def active_role(self) -> str:
        if self.ppo.attacker_controller == "planner":
            return "interceptor"
        if self.update_index < self.ppo.bootstrap_updates:
            return "attacker" if self.update_index % 2 == 0 else "interceptor"
        phase = (self.update_index - self.ppo.bootstrap_updates) // self.ppo.phase_updates
        return "attacker" if phase % 2 == 0 else "interceptor"

    def train(self, updates: int, *, max_seconds: float | None = None) -> list[dict[str, float | int | str]]:
        deadline = None if max_seconds is None else time.perf_counter() + max_seconds
        output: list[dict[str, float | int | str]] = []
        for _ in range(updates):
            role = self.active_role()
            started = time.perf_counter()
            opponent_entry, opponent = self._sample_historical_opponent(role)
            rollout_started = time.perf_counter()
            rollout = self.collect_rollout(
                scripted_interceptor=self.update_index < self.ppo.bootstrap_updates and role == "attacker",
                scripted_attacker=self.update_index < self.ppo.bootstrap_updates and role == "interceptor",
                opponent_role=("interceptor" if role == "attacker" else "attacker") if opponent else None,
                opponent=opponent,
            )
            self._synchronize_device()
            rollout_seconds = time.perf_counter() - rollout_started
            update_started = time.perf_counter()
            losses = self.update_policy(role, rollout)
            self._synchronize_device()
            update_seconds = time.perf_counter() - update_started
            rating_change = self._rate_historical_match(role, opponent_entry, rollout["outcomes"])
            elapsed = time.perf_counter() - started
            transitions = self.ppo.rollout_steps * self.simulator_config.num_envs
            metrics: dict[str, float | int | str] = {
                "update": self.update_index,
                "role": role,
                "attacker_controller": self.ppo.attacker_controller,
                "environment_steps": transitions,
                "steps_per_second": round(transitions / max(elapsed, 1.0e-9), 1),
                "attacker_reward_mean": float(rollout["attacker_rewards"].mean()),
                "interceptor_reward_mean": float(rollout["interceptor_rewards"].mean()),
                "terminal_rate": float(rollout["dones"].mean()),
                "rollout_seconds": round(rollout_seconds, 4),
                "update_seconds": round(update_seconds, 4),
                "inference_seconds": round(self.last_rollout_timing["inference_seconds"], 4),
                "simulation_seconds": round(self.last_rollout_timing["simulation_seconds"], 4),
                "rating": round(self.current_rating[role], 2),
                "rating_change": round(rating_change, 3),
                "opponent_update": opponent_entry.update if opponent_entry else -1,
                **losses,
            }
            metrics.update(self._rollout_metrics(rollout))
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")
            output.append(metrics)
            self.update_index += 1
            if self.update_index % self.ppo.snapshot_interval == 0:
                self.save_checkpoint(self.checkpoint_directory / f"update-{self.update_index:06d}.pt")
            if self.stop_requested or (deadline is not None and time.perf_counter() >= deadline):
                break
        return output

    def _synchronize_device(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _rollout_metrics(self, rollout: dict[str, np.ndarray]) -> dict[str, float | int]:
        outcomes = rollout["outcomes"].reshape(-1)
        completed = outcomes != 0
        goals = outcomes == 2
        captures = outcomes == 1
        timeouts = outcomes == 3
        completed_detected = rollout["detected"].reshape(-1)[completed]
        attacker_rewards = rollout["attacker_rewards"].reshape(-1)
        interceptor_rewards = rollout["interceptor_rewards"].reshape(-1)
        def quantiles(values: np.ndarray, prefix: str) -> dict[str, float]:
            result = np.quantile(values, [0.05, 0.50, 0.95])
            return {
                f"{prefix}_p05": float(result[0]),
                f"{prefix}_p50": float(result[1]),
                f"{prefix}_p95": float(result[2]),
            }
        episode_ticks = rollout["elapsed_ticks"].reshape(-1)[completed]
        result: dict[str, float | int] = {
            "capture_count": int(captures.sum()),
            "goal_count": int(goals.sum()),
            "timeout_count": int(timeouts.sum()),
            "clean_goal_count": int(np.logical_and(goals, ~rollout["detected"].reshape(-1).astype(bool)).sum()),
            "detected_terminal_rate": float(completed_detected.mean()) if completed_detected.size else 0.0,
            "episode_ticks_mean": float(episode_ticks.mean()) if episode_ticks.size else 0.0,
            "episode_ticks_p95": float(np.quantile(episode_ticks, 0.95)) if episode_ticks.size else 0.0,
            "shaping_reward_mean": float(rollout["shaping_rewards"].mean()),
            "detection_reward_mean": float(rollout["detection_rewards"].mean()),
            "terminal_reward_mean": float(rollout["terminal_rewards"].mean()),
            "camp_reward_mean": float(rollout["camp_rewards"].mean()),
            "tracking_reward_mean": float(rollout["tracking_rewards"].mean()),
            "closing_reward_mean": float(rollout["closing_rewards"].mean()),
        }
        result.update(quantiles(attacker_rewards, "attacker_reward"))
        result.update(quantiles(interceptor_rewards, "interceptor_reward"))
        return result

    def _sample_historical_opponent(self, active_role: str):
        if self.ppo.attacker_controller == "planner":
            return None, None
        if self.update_index < self.ppo.bootstrap_updates:
            return None, None
        if random.random() >= self.ppo.historical_opponent_probability:
            return None, None
        opponent_role = "interceptor" if active_role == "attacker" else "attacker"
        entry = self.league.sample(opponent_role, random)
        if entry is None:
            return None, None
        path = self.run_directory / entry.checkpoint
        if not path.is_file():
            return None, None
        payload = self.torch.load(path, map_location=self.device, weights_only=False)
        config = PPOConfig(**payload["ppo"])
        model = make_policy(policy_specs()[opponent_role], config.architecture, config.spike_steps).to(self.device)
        model.load_state_dict(payload[opponent_role])
        model.eval()
        return entry, model

    def _rate_historical_match(self, active_role: str, opponent_entry, outcomes: np.ndarray) -> float:
        if opponent_entry is None:
            return 0.0
        completed = outcomes[outcomes != 0]
        if completed.size == 0:
            return 0.0
        if active_role == "attacker":
            active_score = float(np.mean(completed == 2))
        else:
            active_score = float(np.mean((completed == 1) | (completed == 3)))
        active_old = self.current_rating[active_role]
        opponent_old = opponent_entry.rating
        active_expected = 1.0 / (1.0 + 10.0 ** ((opponent_old - active_old) / 400.0))
        self.current_rating[active_role] += 32.0 * (active_score - active_expected)
        self.league.update_rating(opponent_entry, 1.0 - active_score, active_old)
        return self.current_rating[active_role] - active_old

    def collect_rollout(
        self, *, scripted_interceptor: bool, scripted_attacker: bool,
        opponent_role: str | None = None, opponent=None,
    ) -> dict[str, np.ndarray]:
        torch = self.torch
        storage: dict[str, list[np.ndarray]] = {
            key: []
            for key in (
                "interceptor_obs", "attacker_obs", "masks", "interceptor_actions", "attacker_actions",
                "interceptor_log_probs", "attacker_log_probs", "interceptor_values", "attacker_values",
                "interceptor_rewards", "attacker_rewards", "dones",
                "outcomes",
                "detected", "elapsed_ticks", "shaping_rewards", "detection_rewards", "terminal_rewards", "camp_rewards",
                "tracking_rewards", "closing_rewards",
            )
        }
        rollout_started = time.perf_counter()
        inference_seconds = 0.0
        simulation_seconds = 0.0
        for _ in range(self.ppo.rollout_steps):
            interceptor_obs, attacker_obs, masks = self.env.observations
            iobs_tensor = torch.as_tensor(interceptor_obs, device=self.device)
            aobs_tensor = torch.as_tensor(attacker_obs, device=self.device)
            inference_started = time.perf_counter()
            with torch.no_grad():
                flat_iobs = iobs_tensor.reshape(-1, iobs_tensor.shape[-1])
                if scripted_interceptor:
                    interceptor_actions = orbit_interceptor_actions(self.env)
                    interceptor_log_probs = np.zeros((self.simulator_config.num_envs, MAX_DEFENDERS), np.float32)
                else:
                    interceptor_actor = opponent if opponent_role == "interceptor" else self.interceptor
                    ia, ilog = interceptor_actor.actor.sample(flat_iobs)
                    interceptor_actions = ia.reshape(self.simulator_config.num_envs, MAX_DEFENDERS, 2).cpu().numpy()
                    interceptor_log_probs = ilog.reshape(self.simulator_config.num_envs, MAX_DEFENDERS).cpu().numpy()
                    interceptor_actions *= masks[..., None]
                if self.ppo.attacker_controller == "planner":
                    attacker_actions = planning_attacker_actions(attacker_obs)
                    attacker_log_probs = np.zeros(self.simulator_config.num_envs, np.float32)
                elif scripted_attacker:
                    attacker_actions = straight_attacker_actions(self.simulator_config.num_envs)
                    attacker_log_probs = np.zeros(self.simulator_config.num_envs, np.float32)
                else:
                    attacker_actor = opponent if opponent_role == "attacker" else self.attacker
                    aa, alog = attacker_actor.actor.sample(aobs_tensor)
                    attacker_actions = aa.cpu().numpy()
                    attacker_log_probs = alog.cpu().numpy()
                interceptor_values = self.interceptor.value(aobs_tensor).cpu().numpy()
                attacker_values = self.attacker.value(aobs_tensor).cpu().numpy()
            inference_seconds += time.perf_counter() - inference_started

            storage["interceptor_obs"].append(interceptor_obs.astype(np.float16, copy=True))
            storage["attacker_obs"].append(attacker_obs.astype(np.float16, copy=True))
            storage["masks"].append(masks.copy())
            storage["interceptor_actions"].append(interceptor_actions.astype(np.float16, copy=False))
            storage["attacker_actions"].append(attacker_actions.astype(np.float16, copy=False))
            storage["interceptor_log_probs"].append(interceptor_log_probs.astype(np.float32, copy=False))
            storage["attacker_log_probs"].append(attacker_log_probs.astype(np.float32, copy=False))
            storage["interceptor_values"].append(interceptor_values.astype(np.float32, copy=False))
            storage["attacker_values"].append(attacker_values.astype(np.float32, copy=False))

            simulation_started = time.perf_counter()
            result = self.env.step(interceptor_actions, attacker_actions)
            simulation_seconds += time.perf_counter() - simulation_started
            done = np.asarray(result["terminated"] | result["truncated"], dtype=np.uint8)
            storage["interceptor_rewards"].append(result["interceptor_reward"].copy())
            storage["attacker_rewards"].append(result["attacker_reward"].copy())
            storage["dones"].append(done)
            storage["outcomes"].append(result["outcome"].copy())
            storage["detected"].append(result["detected"].copy())
            storage["elapsed_ticks"].append(result["elapsed_ticks"].copy())
            storage["shaping_rewards"].append(result["shaping_reward"].copy())
            storage["detection_rewards"].append(result["detection_reward"].copy())
            storage["terminal_rewards"].append(result["terminal_reward"].copy())
            storage["camp_rewards"].append(result["camp_reward"].copy())
            storage["tracking_rewards"].append(result["tracking_reward"].copy())
            storage["closing_rewards"].append(result["closing_reward"].copy())
            if np.any(done):
                self.env.reset_done()

        with torch.no_grad():
            final_global = torch.as_tensor(self.env.arrays["attacker_obs"], device=self.device)
            last_interceptor_value = self.interceptor.value(final_global).cpu().numpy()
            last_attacker_value = self.attacker.value(final_global).cpu().numpy()
        rollout = {key: np.stack(value) for key, value in storage.items()}
        self.last_rollout_timing = {
            "rollout_seconds": time.perf_counter() - rollout_started,
            "inference_seconds": inference_seconds,
            "simulation_seconds": simulation_seconds,
        }
        rollout["interceptor_advantages"], rollout["interceptor_returns"] = self._gae(
            rollout["interceptor_rewards"], rollout["interceptor_values"], rollout["dones"],
            last_interceptor_value,
        )
        rollout["attacker_advantages"], rollout["attacker_returns"] = self._gae(
            rollout["attacker_rewards"], rollout["attacker_values"], rollout["dones"],
            last_attacker_value,
        )
        return rollout

    def _gae(self, rewards, values, dones, last_value) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        running = np.zeros(rewards.shape[1], dtype=np.float32)
        next_value = last_value
        for step in range(rewards.shape[0] - 1, -1, -1):
            continuation = 1.0 - dones[step].astype(np.float32)
            delta = rewards[step] + self.ppo.gamma * next_value * continuation - values[step]
            running = delta + self.ppo.gamma * self.ppo.gae_lambda * continuation * running
            advantages[step] = running
            next_value = values[step]
        return advantages, advantages + values

    def update_policy(self, role: str, rollout: dict[str, np.ndarray]) -> dict[str, float]:
        if role == "attacker":
            return self._update_attacker(rollout)
        return self._update_interceptor(rollout)

    def _update_attacker(self, data: dict[str, np.ndarray]) -> dict[str, float]:
        torch = self.torch
        obs = torch.as_tensor(data["attacker_obs"].reshape(-1, ATTACKER_OBS_DIM), device=self.device).float()
        actions = torch.as_tensor(data["attacker_actions"].reshape(-1, 2), device=self.device).float()
        old_log = torch.as_tensor(data["attacker_log_probs"].reshape(-1), device=self.device)
        advantages = torch.as_tensor(data["attacker_advantages"].reshape(-1), device=self.device)
        returns = torch.as_tensor(data["attacker_returns"].reshape(-1), device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
        return self._ppo_epochs(self.attacker, self.attacker_optimizer, obs, obs, actions, old_log, advantages, returns)

    def _update_interceptor(self, data: dict[str, np.ndarray]) -> dict[str, float]:
        torch = self.torch
        shape = data["interceptor_obs"].shape
        actor_obs = data["interceptor_obs"].reshape(-1, shape[-1])
        actions = data["interceptor_actions"].reshape(-1, 2)
        old_log = data["interceptor_log_probs"].reshape(-1)
        masks = data["masks"].reshape(-1).astype(bool)
        team_advantages = np.repeat(data["interceptor_advantages"][..., None], MAX_DEFENDERS, axis=2).reshape(-1)
        active = np.flatnonzero(masks)
        actor_obs_t = torch.as_tensor(actor_obs[active], device=self.device).float()
        actions_t = torch.as_tensor(actions[active], device=self.device).float()
        old_log_t = torch.as_tensor(old_log[active], device=self.device)
        advantages_t = torch.as_tensor(team_advantages[active], device=self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1.0e-8)
        global_obs = torch.as_tensor(data["attacker_obs"].reshape(-1, ATTACKER_OBS_DIM), device=self.device).float()
        returns = torch.as_tensor(data["interceptor_returns"].reshape(-1), device=self.device)
        return self._ppo_epochs(
            self.interceptor, self.interceptor_optimizer, actor_obs_t, global_obs, actions_t,
            old_log_t, advantages_t, returns, separate_critic=True,
        )

    def _ppo_epochs(
        self, model, optimizer, actor_obs, critic_obs, actions, old_log, advantages, returns,
        *, separate_critic: bool = False,
    ) -> dict[str, float]:
        torch = self.torch
        actor_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        batch = self.ppo.minibatch_size
        for _ in range(self.ppo.epochs):
            actor_order = torch.randperm(actor_obs.shape[0], device=self.device)
            for start in range(0, actor_obs.shape[0], batch):
                index = actor_order[start : start + batch]
                log_prob, entropy = model.actor.evaluate(actor_obs[index], actions[index])
                ratio = (log_prob - old_log[index]).exp()
                unclipped = ratio * advantages[index]
                clipped = ratio.clamp(1.0 - self.ppo.clip, 1.0 + self.ppo.clip) * advantages[index]
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                loss = actor_loss - self.ppo.entropy_coefficient * entropy.mean()
                if not separate_critic:
                    value_loss = (model.value(critic_obs[index]) - returns[index]).square().mean()
                    loss = loss + self.ppo.value_coefficient * value_loss
                    value_losses.append(float(value_loss.detach().cpu()))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.ppo.max_grad_norm)
                optimizer.step()
                actor_losses.append(float(actor_loss.detach().cpu()))
                entropies.append(float(entropy.mean().detach().cpu()))
            if separate_critic:
                critic_order = torch.randperm(critic_obs.shape[0], device=self.device)
                for start in range(0, critic_obs.shape[0], batch):
                    index = critic_order[start : start + batch]
                    value_loss = (model.value(critic_obs[index]) - returns[index]).square().mean()
                    optimizer.zero_grad(set_to_none=True)
                    (self.ppo.value_coefficient * value_loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.ppo.max_grad_norm)
                    optimizer.step()
                    value_losses.append(float(value_loss.detach().cpu()))
        return {
            "actor_loss": float(np.mean(actor_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
        }

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "update": self.update_index,
            "simulator": asdict(self.simulator_config),
            "ppo": asdict(self.ppo),
            "interceptor": self.interceptor.state_dict(),
            "attacker": self.attacker.state_dict(),
            "interceptor_optimizer": self.interceptor_optimizer.state_dict(),
            "attacker_optimizer": self.attacker_optimizer.state_dict(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "torch_rng": self.torch.get_rng_state(),
            "torch_cuda_rng": self.torch.cuda.get_rng_state_all() if self.torch.cuda.is_available() else None,
            "current_rating": self.current_rating,
        }
        self.torch.save(payload, path)
        try:
            checkpoint_reference = str(path.relative_to(self.run_directory))
        except ValueError:
            checkpoint_reference = str(path.resolve())
        for role in ("attacker", "interceptor"):
            self.league.add(LeagueEntry(
                role, self.update_index, checkpoint_reference, self.current_rating[role]
            ))

    def load_checkpoint(self, path: Path) -> None:
        payload = self.torch.load(path, map_location="cpu", weights_only=False)
        self.interceptor.load_state_dict(payload["interceptor"])
        self.attacker.load_state_dict(payload["attacker"])
        self.interceptor_optimizer.load_state_dict(payload["interceptor_optimizer"])
        self.attacker_optimizer.load_state_dict(payload["attacker_optimizer"])
        self.update_index = int(payload["update"])
        np.random.set_state(payload["numpy_rng"])
        random.setstate(payload["python_rng"])
        self.torch.set_rng_state(payload["torch_rng"])
        if self.torch.cuda.is_available() and payload.get("torch_cuda_rng") is not None:
            self.torch.cuda.set_rng_state_all(payload["torch_cuda_rng"])
        self.current_rating = payload.get("current_rating", self.current_rating)


def evaluate_policies(interceptor, attacker, simulator: SimulatorConfig, episodes: int, device: str, attacker_controller: str = "learned") -> dict[str, Any]:
    torch = require_torch()
    env = VectorEnv(simulator)
    results = {"interceptor_wins": 0, "attacker_goals": 0, "timeouts": 0, "clean_goals": 0, "episodes": 0}
    while results["episodes"] < episodes:
        iobs, aobs, masks = env.observations
        with torch.no_grad():
            iobs_t = torch.as_tensor(iobs, device=device)
            aobs_t = torch.as_tensor(aobs, device=device)
            ia, _ = interceptor.actor.sample(iobs_t.reshape(-1, iobs_t.shape[-1]), deterministic=True)
            if attacker_controller == "planner":
                attacker_actions = planning_attacker_actions(aobs)
            else:
                aa, _ = attacker.actor.sample(aobs_t, deterministic=True)
                attacker_actions = aa.cpu().numpy()
        stepped = env.step(ia.reshape(simulator.num_envs, MAX_DEFENDERS, 2).cpu().numpy(), attacker_actions)
        done = stepped["terminated"] | stepped["truncated"]
        for index in np.flatnonzero(done):
            if results["episodes"] >= episodes:
                break
            outcome = int(stepped["outcome"][index])
            results["interceptor_wins"] += outcome == 1
            results["attacker_goals"] += outcome == 2
            results["timeouts"] += outcome == 3
            results["clean_goals"] += outcome == 2 and not bool(stepped["detected"][index])
            results["episodes"] += 1
        if np.any(done):
            env.reset_done()
    results["attacker_win_rate"] = results["attacker_goals"] / max(1, results["episodes"])
    return results
