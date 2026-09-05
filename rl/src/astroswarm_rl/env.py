from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from . import _native

MAX_DEFENDERS = int(_native.MAX_DEFENDERS)
INTERCEPTOR_OBS_DIM = int(_native.INTERCEPTOR_OBS_DIM)
ATTACKER_OBS_DIM = int(_native.ATTACKER_OBS_DIM)
SELF_DIM = int(_native.SELF_DIM)
INTERCEPTOR_ENTITY_DIM = int(_native.INTERCEPTOR_ENTITY_DIM)
ATTACKER_ENTITY_DIM = int(_native.ATTACKER_ENTITY_DIM)


@dataclass(slots=True)
class SimulatorConfig:
    num_envs: int = 16_384
    threads: int = 8
    profile: Literal["balanced_self_play", "farp_parity"] = "balanced_self_play"
    action_repeat: int = 4
    episode_seconds: float = 30.0
    min_defenders: int = 1
    max_defenders: int = 6
    primary_five_mix: bool = True
    collisions: bool = False
    defender_fov_min_degrees: float = 35.0
    defender_fov_max_degrees: float = 75.0
    defender_camp_radius: float = 180.0
    defender_camp_penalty: float = 0.001
    detection_reward: float = 0.05
    capture_reward: float = 1.0
    tracking_reward: float = 0.0
    closing_reward: float = 0.0
    seed: int = 1


class VectorEnv:
    """Batched native simulator whose NumPy views are overwritten in place."""

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self.config = config or SimulatorConfig()
        self._native = _native.BatchedSimulator(asdict(self.config))
        self._arrays = self._native.arrays()

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        return self._arrays

    @property
    def observations(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self._arrays["interceptor_obs"],
            self._arrays["attacker_obs"],
            self._arrays["defender_mask"],
        )

    @property
    def state(self) -> dict[str, np.ndarray]:
        return self._native.state()

    @property
    def physics_ticks(self) -> int:
        return int(self._native.physics_ticks)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._native.reset(self.config.seed if seed is None else seed)
        return self.observations

    def reset_done(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._native.reset_done()
        return self.observations

    def reset_mask(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        clean = np.ascontiguousarray(mask, dtype=np.uint8).reshape(self.config.num_envs)
        self._native.reset_mask(clean)
        return self.observations

    def step(
        self,
        interceptor_actions: np.ndarray,
        attacker_actions: np.ndarray,
    ) -> dict[str, np.ndarray]:
        interceptors = np.ascontiguousarray(interceptor_actions, dtype=np.float32)
        attacker = np.ascontiguousarray(attacker_actions, dtype=np.float32)
        expected_interceptors = (self.config.num_envs, MAX_DEFENDERS, 2)
        expected_attacker = (self.config.num_envs, 2)
        if interceptors.shape != expected_interceptors:
            raise ValueError(f"interceptor_actions must have shape {expected_interceptors}")
        if attacker.shape != expected_attacker:
            raise ValueError(f"attacker_actions must have shape {expected_attacker}")
        self._native.step(interceptors, attacker)
        return self._arrays

    def sample_actions(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        interceptors = rng.uniform(-1.0, 1.0, (self.config.num_envs, MAX_DEFENDERS, 2)).astype(np.float32)
        attacker = rng.uniform(-1.0, 1.0, (self.config.num_envs, 2)).astype(np.float32)
        interceptors *= self._arrays["defender_mask"][..., None]
        return interceptors, attacker


def straight_attacker_actions(num_envs: int) -> np.ndarray:
    actions = np.zeros((num_envs, 2), dtype=np.float32)
    actions[:, 0] = 1.0
    return actions


def orbit_interceptor_actions(env: VectorEnv, turn: float = -0.625) -> np.ndarray:
    actions = np.zeros((env.config.num_envs, MAX_DEFENDERS, 2), dtype=np.float32)
    actions[..., 0] = 1.0
    actions[..., 1] = turn
    observations = env.arrays["interceptor_obs"]
    enemy = observations[..., SELF_DIM : SELF_DIM + INTERCEPTOR_ENTITY_DIM]
    visible = enemy[..., 10] > 0.5
    bearing_sin = enemy[..., 7]
    actions[..., 1] = np.where(visible, np.clip(bearing_sin * 2.0, -1.0, 1.0), actions[..., 1])
    actions *= env.arrays["defender_mask"][..., None]
    return actions
