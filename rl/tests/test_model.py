from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from astroswarm_rl.artifacts import export_actor
from astroswarm_rl.model import make_policy, policy_specs


@pytest.mark.parametrize("architecture", ["mlp", "snn"])
@pytest.mark.parametrize("role", ["attacker", "interceptor"])
def test_policy_shapes_and_export(role, architecture, tmp_path) -> None:
    spec = policy_specs()[role]
    policy = make_policy(spec, architecture)
    batch = torch.zeros(7, spec.obs_dim)
    action, log_prob = policy.actor.sample(batch)

    assert action.shape == (7, 2)
    assert log_prob.shape == (7,)
    assert policy.value(torch.zeros(7, spec.global_dim)).shape == (7,)

    path = tmp_path / f"{role}.policy.json"
    payload = export_actor(policy, spec, path, "balanced_self_play")
    loaded = json.loads(path.read_text())
    assert loaded["sha256"] == payload["sha256"]
    assert loaded["role"] == role


@pytest.mark.parametrize("architecture", ["mlp", "snn"])
@pytest.mark.parametrize("role", ["attacker", "interceptor"])
def test_godot_policy_matches_torch(role, architecture, tmp_path) -> None:
    godot = shutil.which("godot")
    if godot is None:
        pytest.skip("Godot is not installed")
    torch.manual_seed(23)
    spec = policy_specs()[role]
    policy = make_policy(spec, architecture).eval()
    observation = torch.randn(1, spec.obs_dim)
    entities = observation[:, spec.self_dim:].reshape(1, spec.entity_count, spec.entity_dim)
    entities[..., spec.mask_index] = torch.arange(spec.entity_count) % 2
    observation_file = tmp_path / "observation.json"
    observation_file.write_text(json.dumps(observation[0].tolist()))
    expected = policy.actor(observation)[0].detach().numpy()
    artifact = tmp_path / f"{role}.policy.json"
    export_actor(policy, spec, artifact, "balanced_self_play")
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["XDG_DATA_HOME"] = str(tmp_path / "godot-data")
    completed = subprocess.run(
        [
            godot, "--headless", "--path", str(repository), "--script",
            "simulations/rl/PolicySmokeTest.gd", "--", str(artifact), str(observation_file),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
        env=environment,
    )
    match = re.search(r'POLICY_RESULT="\(([-+0-9.eE]+), ([-+0-9.eE]+)\)"', completed.stdout)
    assert match is not None, completed.stdout + completed.stderr
    actual = torch.tensor([float(match.group(1)), float(match.group(2))])
    torch.testing.assert_close(actual, torch.as_tensor(expected), atol=2.0e-5, rtol=2.0e-5)


@pytest.mark.parametrize("sensor_bounds", [None, (45.0, 55.0)])
def test_arena_uses_exported_fov_or_legacy_defaults(sensor_bounds, tmp_path) -> None:
    godot = shutil.which("godot")
    if godot is None:
        pytest.skip("Godot is not installed")
    spec = policy_specs()["interceptor"]
    artifact = tmp_path / "interceptor.policy.json"
    simulator = None if sensor_bounds is None else {
        "defender_fov_min_degrees": sensor_bounds[0], "defender_fov_max_degrees": sensor_bounds[1],
    }
    payload = export_actor(make_policy(spec), spec, artifact, "balanced_self_play", simulator=simulator)
    if simulator is not None:
        assert payload["simulator"] == simulator
    script = tmp_path / "check_fov.gd"
    lower, upper = sensor_bounds or (35.0, 75.0)
    script.write_text('''extends SceneTree
func _initialize():
    call_deferred("check_arena")
func check_arena():
    var arena = load("res://simulations/rl/AIArena.gd").new()
    root.add_child(arena)
    arena.defenders[0]["last_attacker"] = Vector4(0.1, 0.2, 0.3, 0.4)
    arena.defenders[0]["last_attacker_age"] = 60
    var observation = arena._interceptor_observation(0)
    if observation.size() != 83 or absf(observation[12] - 0.1) > 0.00001 or absf(observation[16] - 0.5) > 0.00001:
        quit(1)
        return
    for defender in arena.defenders:
        var degrees = rad_to_deg(defender["fov"])
        if degrees < %s - 0.001 or degrees > %s + 0.001:
            quit(1)
            return
    print("FOV_OK")
    quit(0)
''' % (lower, upper))
    environment = os.environ.copy()
    environment["XDG_DATA_HOME"] = str(tmp_path / "godot-data")
    result = subprocess.run([godot, "--headless", "--path", str(Path(__file__).resolve().parents[2]),
                             "--script", str(script), "--", f"--interceptor-policy={artifact}"],
                            capture_output=True, text=True, env=environment, timeout=20, check=True)
    assert "FOV_OK" in result.stdout, result.stdout + result.stderr


@pytest.mark.parametrize("role", ["attacker", "interceptor"])
def test_snn_gradients_and_decision_reset(role):
    torch.manual_seed(31)
    spec = policy_specs()[role]
    model = make_policy(spec, "snn")
    observation = torch.randn(5, spec.obs_dim)
    expected = model.actor(observation).detach()
    model.actor(torch.randn_like(observation))
    torch.testing.assert_close(model.actor(observation), expected, rtol=0, atol=0)
    actions, old_log = model.actor.sample(observation)
    log_prob, _ = model.actor.evaluate(observation, actions.detach())
    torch.testing.assert_close(log_prob, old_log.detach(), atol=1e-4, rtol=1e-4)
    (-log_prob.mean()).backward()
    for layer in (model.actor.self_encoder[0], model.actor.trunk[0], model.actor.trunk[2]):
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum() > 0
