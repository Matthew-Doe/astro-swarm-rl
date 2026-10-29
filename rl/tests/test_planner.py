from dataclasses import asdict
import numpy as np
import pytest

from astroswarm_rl.env import SimulatorConfig, VectorEnv, SELF_DIM, ATTACKER_ENTITY_DIM
from astroswarm_rl.planner import planning_attacker_actions


def test_planner_targets_goal_and_ignores_inactive_defenders():
    env = VectorEnv(SimulatorConfig(num_envs=3, threads=1))
    obs = env.observations[1].copy()
    obs[:, :2] = [0.8, 0.0]
    obs[:, 2:4] = [0.0, -1.0]
    obs[:, SELF_DIM:] = 0
    straight = planning_attacker_actions(obs)
    np.testing.assert_allclose(straight, [[1, 0]] * 3, atol=1e-5)
    entities = obs[:, SELF_DIM:].reshape(3, -1, ATTACKER_ENTITY_DIM)
    entities[..., :2] = 99
    np.testing.assert_array_equal(planning_attacker_actions(obs), straight)
    entities[:, 0, :2] = [-0.2, 0]
    entities[:, 0, 12] = 1
    avoiding = planning_attacker_actions(obs)
    assert np.all(np.abs(avoiding[:, 1]) > 0.1)
    assert np.isfinite(avoiding).all()
    assert np.max(np.abs(avoiding)) <= 1


def test_planner_training_only_updates_interceptors(tmp_path):
    torch = pytest.importorskip('torch')
    from astroswarm_rl.training import PPOConfig, SelfPlayTrainer, evaluate_policies
    trainer = SelfPlayTrainer(SimulatorConfig(num_envs=2, threads=1),
        PPOConfig(architecture='snn', attacker_controller='planner', rollout_steps=4,
                  epochs=1, minibatch_size=16), tmp_path, device='cpu')
    before = {key: value.clone() for key, value in trainer.attacker.state_dict().items()}
    interceptor_before = {key: value.clone() for key, value in trainer.interceptor.state_dict().items()}
    metrics = trainer.train(100, max_seconds=0.001)
    assert len(metrics) == 1
    assert metrics[0]['role'] == 'interceptor'
    assert metrics[0]['attacker_controller'] == 'planner'
    assert trainer._sample_historical_opponent('interceptor') == (None, None)
    for key, value in trainer.attacker.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert any(not torch.equal(value, interceptor_before[key]) for key, value in trainer.interceptor.state_dict().items())
    trainer.save_checkpoint(tmp_path / 'final.pt')
    payload = torch.load(tmp_path / 'final.pt', weights_only=False)
    assert payload['ppo']['attacker_controller'] == 'planner'
    trainer.load_checkpoint(tmp_path / 'final.pt')
    config = SimulatorConfig(num_envs=2, threads=1, episode_seconds=0.2)
    result = evaluate_policies(trainer.interceptor, trainer.attacker, config, 2, 'cpu', 'planner')
    assert result['episodes'] == 2
