from pathlib import Path

import numpy as np
import pytest

from astroswarm_rl.env import MAX_DEFENDERS, SimulatorConfig, VectorEnv


def arena(defenders=2, **kwargs):
    config = dict(num_envs=1, threads=1, profile="farp_parity", min_defenders=defenders,
                  max_defenders=defenders, action_repeat=1, seed=7)
    config.update(kwargs)
    return VectorEnv(SimulatorConfig(**config))


def place(env, *, tracking=False):
    state = env.state
    state["defender_x"][:] = 2800
    state["defender_y"][:] = 1700
    state["defender_heading"][:] = np.pi
    state["defender_x"][0, 0] = 2090
    state["defender_y"][0, 0] = 1080
    state["defender_heading"][0, 0] = 0
    state["attacker_x"][0] = 2220 if tracking else 3000
    state["attacker_y"][0] = 1080 if tracking else 2000


def step(env, ia=None, aa=None):
    result = env.step(np.zeros((1, MAX_DEFENDERS, 2), np.float32) if ia is None else ia,
                      np.zeros((1, 2), np.float32) if aa is None else aa)
    components = sum(result[k] for k in ("camp_reward", "tracking_reward", "closing_reward",
                                         "terminal_reward", "detection_reward", "shaping_reward"))
    np.testing.assert_allclose(result["attacker_reward"], components, atol=1e-6)
    np.testing.assert_allclose(result["attacker_reward"] + result["interceptor_reward"], 0, atol=1e-6)
    return result


def test_camping_grace_fraction_and_tracking_exemption():
    env = arena(defender_camp_conditional=True, defender_camp_grace_seconds=2,
                defender_camp_penalty=.01)
    place(env)
    for _ in range(120):
        assert step(env)["camp_reward"][0] == 0
    assert step(env)["camp_reward"][0] == pytest.approx(.005)  # One of two active defenders.
    place(env, tracking=True)
    assert step(env)["camp_reward"][0] == 0
    place(env)
    for _ in range(120):
        assert step(env)["camp_reward"][0] == 0
    assert step(env)["camp_reward"][0] == pytest.approx(.005)
    env.reset()
    place(env)
    assert step(env)["camp_reward"][0] == 0


def test_movement_restarts_camping_grace():
    env = arena(defender_camp_conditional=True, defender_camp_grace_seconds=0.05,
                defender_camp_penalty=.01)
    place(env)
    for _ in range(4):
        result = step(env)
    assert result["camp_reward"][0] > 0
    ia = np.zeros((1, MAX_DEFENDERS, 2), np.float32)
    ia[0, 0, 0] = .5
    assert step(env, ia=ia)["camp_reward"][0] == 0
    assert step(env)["camp_reward"][0] == 0


@pytest.mark.parametrize("defenders", [1, 5])
def test_team_tracking_cap_and_episode_reset(defenders):
    env = arena(defenders=defenders, tracking_team_contact=True,
                tracking_reward=.003, tracking_reward_cap=.007)
    place(env, tracking=True)
    for expected in (.003, .003, .001, 0):
        assert -step(env)["tracking_reward"][0] == pytest.approx(expected, abs=1e-8)
    env.reset()
    place(env, tracking=True)
    assert -step(env)["tracking_reward"][0] == pytest.approx(.003)


def test_team_tracking_has_no_extra_payment_for_redundancy():
    env = arena(defenders=5, tracking_team_contact=True, tracking_reward=.003)
    place(env, tracking=True)
    one = float(step(env)["tracking_reward"][0])
    env.state["defender_x"][:] = 2090
    env.state["defender_y"][:] = 1080
    env.state["defender_heading"][:] = 0
    assert float(step(env)["tracking_reward"][0]) == one


def test_old_tracking_mode_still_uses_defender_fraction():
    env = arena(defenders=5, tracking_reward=.003)
    place(env, tracking=True)
    assert -step(env)["tracking_reward"][0] == pytest.approx(.003 / 5)


def pursuit_arena(**kwargs):
    env = arena(defenders=1, action_repeat=4, closing_reward=.5,
                closing_potential_shaping=True, **kwargs)
    env.state["defender_x"][0, 0] = 2220
    env.state["defender_y"][0, 0] = 1080
    env.state["attacker_x"][0] = 2520
    env.state["attacker_y"][0] = 1080
    return env


def test_pursuit_rewards_progress_and_cannot_farm_round_trip():
    env = pursuit_arena()
    env.state["attacker_heading"][0] = np.pi
    aa = np.array([[1, 0]], np.float32)
    approach = -float(step(env, aa=aa)["closing_reward"][0])
    assert approach > 0
    env.state["attacker_heading"][0] = 0
    retreat = -float(step(env, aa=aa)["closing_reward"][0])
    assert retreat < 0
    gamma = env.config.reward_shaping_gamma
    assert approach + gamma * retreat == pytest.approx(.5 * (gamma**2 * .7 - .7), abs=1e-7)


def test_terminal_potential_is_zero_and_done_rewards_stay_zero():
    env = pursuit_arena(episode_seconds=.001, timeout_reward=.25)
    result = step(env)
    assert result["outcome"][0] == 3
    assert result["terminal_reward"][0] == -.25
    assert result["closing_reward"][0] == pytest.approx(.5 * .7)
    assert step(env)["closing_reward"][0] == 0


def test_legacy_closing_also_penalizes_retreat():
    env = pursuit_arena()
    # Construct the same geometry in legacy delta mode.
    legacy = arena(defenders=1, action_repeat=4, closing_reward=.5)
    for key in legacy.state:
        legacy.state[key][:] = env.state[key]
    legacy.state["attacker_heading"][0] = 0
    assert step(legacy, aa=np.array([[1, 0]], np.float32))["closing_reward"][0] > 0


def test_new_preset_and_discount_validation(tmp_path):
    from astroswarm_rl.cli import _load_config
    from astroswarm_rl.training import SelfPlayTrainer
    config, ppo = _load_config(Path(__file__).parents[1] / "configs/reward_defense.toml")
    assert config.capture_reward == 3
    assert config.clean_goal_reward == 5
    assert config.tracking_reward_cap == .3
    assert (config.defender_fov_min_degrees, config.defender_fov_max_degrees) == (35, 75)
    assert config.reward_shaping_gamma == ppo.gamma
    config.reward_shaping_gamma = .9
    with pytest.raises(ValueError, match="must match PPO"):
        SelfPlayTrainer(config, ppo, tmp_path, device="cpu")
