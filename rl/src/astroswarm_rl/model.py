from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional system package
        raise RuntimeError(
            "PyTorch is not installed. Run rl/scripts/bootstrap-arch.sh to install the ROCm build."
        ) from exc
    return torch


def _classes():
    torch = require_torch()
    nn = torch.nn

    class EntityActor(nn.Module):
        def __init__(self, obs_dim: int, self_dim: int, entity_count: int, entity_dim: int, mask_index: int) -> None:
            super().__init__()
            self.obs_dim = obs_dim
            self.self_dim = self_dim
            self.entity_count = entity_count
            self.entity_dim = entity_dim
            self.mask_index = mask_index
            self.self_encoder = nn.Sequential(nn.Linear(self_dim, 64), nn.Tanh())
            self.entity_encoder = nn.Sequential(
                nn.Linear(entity_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh()
            )
            self.trunk = nn.Sequential(
                nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh()
            )
            self.mean = nn.Linear(128, 2)
            self.log_std = nn.Parameter(torch.full((2,), -0.5))

        def encode(self, obs):
            self_features = obs[..., : self.self_dim]
            entities = obs[..., self.self_dim :].reshape(
                *obs.shape[:-1], self.entity_count, self.entity_dim
            )
            mask = entities[..., self.mask_index : self.mask_index + 1].clamp(0.0, 1.0)
            encoded_entities = self.entity_encoder(entities) * mask
            denominator = mask.sum(dim=-2).clamp_min(1.0)
            pooled = encoded_entities.sum(dim=-2) / denominator
            return self.trunk(torch.cat((self.self_encoder(self_features), pooled), dim=-1))

        def forward(self, obs):
            return torch.tanh(self.mean(self.encode(obs)))

        def distribution(self, obs):
            mean = self.mean(self.encode(obs))
            std = self.log_std.exp().expand_as(mean)
            return torch.distributions.Normal(mean, std)

        def sample(self, obs, deterministic: bool = False):
            distribution = self.distribution(obs)
            raw = distribution.mean if deterministic else distribution.rsample()
            action = torch.tanh(raw)
            log_prob = distribution.log_prob(raw).sum(-1)
            log_prob -= torch.log(1.0 - action.square() + 1.0e-6).sum(-1)
            return action, log_prob

        def evaluate(self, obs, action):
            clipped = action.clamp(-0.999999, 0.999999)
            raw = torch.atanh(clipped)
            distribution = self.distribution(obs)
            log_prob = distribution.log_prob(raw).sum(-1)
            log_prob -= torch.log(1.0 - clipped.square() + 1.0e-6).sum(-1)
            return log_prob, distribution.entropy().sum(-1)

    class ActorCritic(nn.Module):
        def __init__(self, actor: EntityActor, global_dim: int) -> None:
            super().__init__()
            self.actor = actor
            self.critic = nn.Sequential(
                nn.Linear(global_dim, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1)
            )

        def value(self, global_obs):
            return self.critic(global_obs).squeeze(-1)

    return EntityActor, ActorCritic


@dataclass(frozen=True)
class PolicySpec:
    role: str
    obs_dim: int
    self_dim: int
    entity_count: int
    entity_dim: int
    mask_index: int
    global_dim: int


def make_policy(spec: PolicySpec):
    EntityActor, ActorCritic = _classes()
    return ActorCritic(
        EntityActor(spec.obs_dim, spec.self_dim, spec.entity_count, spec.entity_dim, spec.mask_index),
        spec.global_dim,
    )


def policy_specs() -> dict[str, PolicySpec]:
    from .env import (
        ATTACKER_ENTITY_DIM,
        ATTACKER_OBS_DIM,
        INTERCEPTOR_SELF_DIM,
        INTERCEPTOR_ENTITY_DIM,
        INTERCEPTOR_OBS_DIM,
        MAX_DEFENDERS,
        SELF_DIM,
    )

    return {
        "interceptor": PolicySpec(
            "interceptor", INTERCEPTOR_OBS_DIM, INTERCEPTOR_SELF_DIM, MAX_DEFENDERS,
            INTERCEPTOR_ENTITY_DIM, 10, ATTACKER_OBS_DIM,
        ),
        "attacker": PolicySpec(
            "attacker", ATTACKER_OBS_DIM, SELF_DIM, MAX_DEFENDERS,
            ATTACKER_ENTITY_DIM, 12, ATTACKER_OBS_DIM,
        ),
    }
