"""Deterministic receding-horizon attacker; no learned parameters."""
from __future__ import annotations

import numpy as np

from .env import ATTACKER_ENTITY_DIM, MAX_DEFENDERS, SELF_DIM


def planning_attacker_actions(observation: np.ndarray) -> np.ndarray:
    """Score nine bounded-turn trajectories over two seconds and replan each step.

    Uses the same global defender observations available to the learned attacker.
    Opponents continue at observed velocity; attacker speed/turn are approximated
    by the profile midpoints (150 units/s and 2.8 radians/s).
    """
    obs = np.asarray(observation, dtype=np.float32)
    entities = obs[:, SELF_DIM:].reshape(-1, MAX_DEFENDERS, ATTACKER_ENTITY_DIM)
    position = obs[:, :2]
    heading = np.arctan2(obs[:, 2], obs[:, 3])
    goal_heading = np.arctan2(-position[:, 1], -position[:, 0])
    error = np.arctan2(np.sin(goal_heading - heading), np.cos(goal_heading - heading))
    offsets = np.linspace(-1.6, 1.6, 9, dtype=np.float32)
    targets = heading[:, None] + error[:, None] + offsets
    angles = np.broadcast_to(heading[:, None], targets.shape).copy()
    points = np.broadcast_to(position[:, None, :], (*targets.shape, 2)).copy()
    # Relative velocity observations are normalized by 400; positions by 1000.
    old_speed = np.maximum(obs[:, 10], 0.0) * 0.15
    attacker_velocity = np.stack((np.cos(heading), np.sin(heading)), axis=-1) * old_speed[:, None]
    defenders = position[:, None, :] + entities[..., :2]
    velocity = entities[..., 2:4] * 0.4 + attacker_velocity[:, None, :]
    active = entities[..., 12] > 0.5
    risk = np.zeros_like(angles)
    boundary = np.zeros_like(angles)
    for step in range(1, 9):
        delta = np.arctan2(np.sin(targets - angles), np.cos(targets - angles))
        angles += np.clip(delta, -2.8 * 0.25, 2.8 * 0.25)
        points += 0.15 * 0.25 * np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        predicted = defenders + velocity * (step * 0.25)
        distances2 = ((points[:, :, None, :] - predicted[:, None, :, :]) ** 2).sum(-1)
        risk += np.where(active[:, None, :], np.exp(-distances2 / 0.02), 0.0).sum(-1)
        boundary += (np.abs(points[..., 0]) > 1.90) | (np.abs(points[..., 1]) > 1.06)
    cost = np.linalg.norm(points, axis=-1) + 0.12 * risk + boundary
    chosen = np.argmin(cost, axis=1)
    desired = error + offsets[chosen]
    desired = np.arctan2(np.sin(desired), np.cos(desired))
    result = np.ones((len(obs), 2), dtype=np.float32)
    result[:, 1] = np.clip(desired * 2.0, -1.0, 1.0)
    return result
