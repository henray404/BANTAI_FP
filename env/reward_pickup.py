"""Pure-tensor staged pickup reward + delivery termination. No Isaac import (unit-testable).

Functions are duck-typed on a runtime env that exposes these buffers (populated by
WarehouseRLEnv / WarehouseGymEnv each step):
    env.ee_pos (N,3), env.box_pos (N,3), env.holding (N,) bool, env.goal_pos (N,3),
    env.grasp_event (N,) bool, env.drop_event (N,) bool.
They only use torch, so they work with any object carrying those attributes.
"""

from __future__ import annotations

import torch

DELIVER_RADIUS_M = 1.5  # box within this xy-distance of the goal zone center = delivered


def approach_box_distance(env) -> torch.Tensor:
    """Phase A dense: distance(ee, box), zero while holding (use with negative weight)."""
    d = torch.norm(env.ee_pos - env.box_pos, dim=-1)
    return torch.where(env.holding, torch.zeros_like(d), d)


def carry_distance(env) -> torch.Tensor:
    """Phase B dense: xy-distance(box, goal), zero while NOT holding (use with negative weight)."""
    d = torch.norm(env.box_pos[:, :2] - env.goal_pos[:, :2], dim=-1)
    return torch.where(env.holding, d, torch.zeros_like(d))


def grasp_success_reward(env) -> torch.Tensor:
    """+1 on the step grasp succeeds (one-shot). Use with positive weight (e.g. 5.0)."""
    return env.grasp_event.float()


def drop_penalty(env) -> torch.Tensor:
    """+1 on the step the box is dropped outside a zone (one-shot). Use with negative weight."""
    return env.drop_event.float()


def pickup_delivered(env) -> torch.Tensor:
    """(N,) bool: holding AND box xy within DELIVER_RADIUS_M of the goal zone center."""
    in_zone = torch.norm(env.box_pos[:, :2] - env.goal_pos[:, :2], dim=-1) < DELIVER_RADIUS_M
    return env.holding & in_zone


def pickup_delivered_reward(env) -> torch.Tensor:
    """+1 per step while the held box is delivered in its zone (float of pickup_delivered)."""
    return pickup_delivered(env).float()
