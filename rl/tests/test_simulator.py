from __future__ import annotations

import numpy as np
import pytest

from astroswarm_rl.env import (
    ATTACKER_OBS_DIM,
    INTERCEPTOR_OBS_DIM,
    MAX_DEFENDERS,
    SimulatorConfig,
    VectorEnv,
    straight_attacker_actions,
)


def make_env(**overrides) -> VectorEnv:
    values = {
        "num_envs": 32,
        "threads": 2,
        "profile": "farp_parity",
        "action_repeat": 1,
        "episode_seconds": 20.0,
        "min_defenders": 5,
        "max_defenders": 5,
        "primary_five_mix": False,
        "seed": 7,
    }
    values.update(overrides)
    return VectorEnv(SimulatorConfig(**values))


def test_shapes_and_masks() -> None:
    env = make_env()
    interceptor_obs, attacker_obs, mask = env.observations

    assert interceptor_obs.shape == (32, MAX_DEFENDERS, INTERCEPTOR_OBS_DIM)
    assert attacker_obs.shape == (32, ATTACKER_OBS_DIM)
    assert mask.shape == (32, MAX_DEFENDERS)
    assert np.all(mask[:, :5] == 1)
    assert np.all(mask[:, 5] == 0)


def test_reset_is_deterministic() -> None:
    first = make_env()
    second = make_env()

    for key in first.state:
        np.testing.assert_array_equal(first.state[key], second.state[key])


def test_straight_attacker_reaches_a_terminal_outcome() -> None:
    env = make_env(num_envs=8)
    interceptors = np.zeros((8, MAX_DEFENDERS, 2), dtype=np.float32)
    attacker = straight_attacker_actions(8)

    for _ in range(700):
        result = env.step(interceptors, attacker)
        if np.all(result["terminated"] | result["truncated"]):
            break

    assert np.all(result["terminated"] == 1)
    assert np.all(np.isin(result["outcome"], [1, 2]))
    np.testing.assert_allclose(result["attacker_reward"] + result["interceptor_reward"], 0.0, atol=1e-6)


def test_action_shapes_are_checked() -> None:
    env = make_env(num_envs=2)
    with pytest.raises(ValueError, match="interceptor_actions"):
        env.step(np.zeros((2, 2), np.float32), np.zeros((2, 2), np.float32))


def test_done_environments_can_be_reset_in_place() -> None:
    env = make_env(num_envs=4, episode_seconds=0.05)
    zeros_i = np.zeros((4, MAX_DEFENDERS, 2), np.float32)
    zeros_a = np.zeros((4, 2), np.float32)
    for _ in range(4):
        env.step(zeros_i, zeros_a)
    assert np.all(env.arrays["truncated"] == 1)

    env.reset_done()

    assert np.all(env.arrays["terminated"] == 0)
    assert np.all(env.arrays["truncated"] == 0)
    assert np.all(env.arrays["outcome"] == 0)


def test_balanced_profile_reduces_fov_and_excludes_planet_interior() -> None:
    env = make_env(profile="balanced_self_play", min_defenders=6, max_defenders=6, primary_five_mix=False)
    attacker_obs = env.arrays["attacker_obs"]
    fov = attacker_obs[:, 12 + 10 :: 13]

    assert np.all(fov >= 35.0 / 180.0 - 1.0e-5)
    assert np.all(fov <= 75.0 / 180.0 + 1.0e-5)
    state = env.state
    distances = np.hypot(state["defender_x"] - 1920.0, state["defender_y"] - 1080.0)
    assert np.all(distances >= 129.0 - 1.0e-4)


def test_camp_penalty_is_exposed_and_zero_sum() -> None:
    env = make_env(defender_camp_radius=10_000.0, defender_camp_penalty=0.25)
    zeros_i = np.zeros((32, MAX_DEFENDERS, 2), dtype=np.float32)
    zeros_a = np.zeros((32, 2), dtype=np.float32)
    result = env.step(zeros_i, zeros_a)

    assert np.all(result["camp_reward"] == 0.25)
    np.testing.assert_allclose(
        result["attacker_reward"] + result["interceptor_reward"], 0.0, atol=1.0e-6
    )
