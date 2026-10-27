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


@pytest.mark.parametrize("detection", ["never", "earlier", "goal_tick"])
@pytest.mark.parametrize("clean_reward", [1.1, 5.0])
def test_clean_goal_reward_requires_never_being_detected(detection, clean_reward) -> None:
    env = make_env(num_envs=1, min_defenders=1, max_defenders=1,
                   clean_goal_reward=clean_reward, tracking_reward=0.003,
                   defender_camp_penalty=0.01)
    state = env.state
    state["attacker_x"][0] = 1920.0 + 135.0
    state["attacker_y"][0] = 1080.0
    state["defender_x"][0, 0] = 1920.0 + 250.0
    state["defender_y"][0, 0] = 1080.0
    # Face away unless testing detection on the exact goal tick.
    state["defender_heading"][0, 0] = np.pi if detection == "goal_tick" else 0.0
    env.arrays["detected"][0] = int(detection == "earlier")
    result = env.step(np.zeros((1, MAX_DEFENDERS, 2), np.float32), np.zeros((1, 2), np.float32))

    assert result["outcome"][0] == 2
    assert bool(result["detected"][0]) == (detection != "never")
    assert result["terminal_reward"][0] == pytest.approx(clean_reward if detection == "never" else 1.0)
    np.testing.assert_allclose(result["attacker_reward"] + result["interceptor_reward"], 0, atol=1e-6)
    components = sum(result[key] for key in ("shaping_reward", "terminal_reward", "detection_reward",
                                           "camp_reward", "tracking_reward", "closing_reward"))
    np.testing.assert_allclose(result["attacker_reward"], components, atol=1e-6)
    # A finished environment cannot pay the goal bonus again.
    next_result = env.step(np.zeros((1, MAX_DEFENDERS, 2), np.float32), np.zeros((1, 2), np.float32))
    assert next_result["terminal_reward"][0] == 0
    assert next_result["attacker_reward"][0] == 0


def test_clean_goal_reward_does_not_change_capture_or_timeout() -> None:
    for timeout in (False, True):
        env = make_env(num_envs=1, min_defenders=1, max_defenders=1,
                       clean_goal_reward=5.0, episode_seconds=0.001 if timeout else 20.0)
        if not timeout:
            env.state["defender_x"][0, 0] = env.state["attacker_x"][0]
            env.state["defender_y"][0, 0] = env.state["attacker_y"][0]
        result = env.step(np.zeros((1, MAX_DEFENDERS, 2), np.float32), np.zeros((1, 2), np.float32))
        assert result["outcome"][0] == (3 if timeout else 1)
        assert result["terminal_reward"][0] == -1.0


def test_combined_stealth_config() -> None:
    from pathlib import Path
    from astroswarm_rl.cli import _load_config
    config, ppo = _load_config(Path(__file__).parents[1] / "configs" / "reward_stealth.toml")
    assert config.tracking_reward == 0.003
    assert config.defender_camp_penalty == 0.01
    assert config.clean_goal_reward == 5.0
    assert (config.defender_fov_min_degrees, config.defender_fov_max_degrees) == (35.0, 75.0)
    assert (ppo.minibatch_size, ppo.epochs) == (65536, 4)
    # Older checkpoint/config dictionaries omit the new setting.
    assert SimulatorConfig().clean_goal_reward == 1.1
    with pytest.raises(ValueError, match="non-negative"):
        make_env(clean_goal_reward=-1)
