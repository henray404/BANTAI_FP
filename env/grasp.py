"""Pure-tensor grasp-success / grasp-lost detection. No Isaac import (unit-testable)."""

from __future__ import annotations

import torch

GRIP_RADIUS_M = 0.08  # EE must be within this of the box center to count as grasping
LIFT_M = 0.05         # box must rise this far above its resting height to count as lifted


def grasp_success(
    ee_pos: torch.Tensor,
    box_pos: torch.Tensor,
    gripper_closed: torch.Tensor,
    box_lift: torch.Tensor,
) -> torch.Tensor:
    """(N,) bool: gripper closed AND EE within GRIP_RADIUS_M of box AND box lifted > LIFT_M."""
    near = torch.norm(ee_pos - box_pos, dim=-1) < GRIP_RADIUS_M
    lifted = box_lift > LIFT_M
    return near & gripper_closed & lifted


def grasp_lost(
    holding: torch.Tensor,
    ee_pos: torch.Tensor,
    box_pos: torch.Tensor,
) -> torch.Tensor:
    """(N,) bool: currently holding but the EE has separated from the box (> 2x GRIP_RADIUS_M)."""
    separated = torch.norm(ee_pos - box_pos, dim=-1) > (2.0 * GRIP_RADIUS_M)
    return holding & separated
