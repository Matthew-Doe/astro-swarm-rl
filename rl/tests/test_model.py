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


@pytest.mark.parametrize("role", ["attacker", "interceptor"])
def test_policy_shapes_and_export(role, tmp_path) -> None:
    spec = policy_specs()[role]
    policy = make_policy(spec)
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


@pytest.mark.parametrize("role", ["attacker", "interceptor"])
def test_godot_policy_matches_torch(role, tmp_path) -> None:
    godot = shutil.which("godot")
    if godot is None:
        pytest.skip("Godot is not installed")
    torch.manual_seed(23)
    spec = policy_specs()[role]
    policy = make_policy(spec).eval()
    observation = torch.zeros(1, spec.obs_dim)
    expected = policy.actor(observation)[0].detach().numpy()
    artifact = tmp_path / f"{role}.policy.json"
    export_actor(policy, spec, artifact, "balanced_self_play")
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["XDG_DATA_HOME"] = str(tmp_path / "godot-data")
    completed = subprocess.run(
        [
            godot, "--headless", "--path", str(repository), "--script",
            "simulations/rl/PolicySmokeTest.gd", "--", str(artifact),
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
