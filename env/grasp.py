"""Pure-tensor grasp-success / grasp-lost detection. No Isaac import (unit-testable)."""

from __future__ import annotations

import torch

# EE must be within this of the box SURFACE (size-aware: box_half is subtracted) to count as a
# contact-grasp. Tuned 2026-06-20 via scripts/tune_arm.py: all 3 categories reach < 0.083 m at
# their best standoff, so 0.15 gives generous "magic-attach" slack (robot stops in front, the box
# welds to the hand without the gripper enclosing it) while staying small enough to avoid latching
# the wrong box — grasp_success only ever checks the COMMANDED target box, so a wider radius is safe.
GRIP_RADIUS_M = 0.25


def grasp_success(
    ee_pos: torch.Tensor,
    box_pos: torch.Tensor,
    gripper_closed: torch.Tensor,
    box_half: torch.Tensor,
) -> torch.Tensor:
    """(N,) bool: gripper closed AND EE within GRIP_RADIUS_M of the box SURFACE (size-aware).

    Boxes (0.21/0.32/0.52 m) are larger than the Franka gripper opening, so we use a
    proximity-to-surface contact model, not physical enclosure/lift. Once true, the env
    kinematically carries the box (see WarehouseRLEnv._carry_held_boxes); lift is a consequence,
    not a precondition. box_half = target box size / 2.
    """
    surface_dist = torch.norm(ee_pos - box_pos, dim=-1) - box_half
    near = surface_dist < GRIP_RADIUS_M
    return near & gripper_closed


def grasp_lost(
    holding: torch.Tensor,
    ee_pos: torch.Tensor,
    box_pos: torch.Tensor,
    box_half: torch.Tensor,
) -> torch.Tensor:
    """(N,) bool: currently holding but the EE has separated from the box surface (> 2x radius)."""
    surface_dist = torch.norm(ee_pos - box_pos, dim=-1) - box_half
    separated = surface_dist > (2.0 * GRIP_RADIUS_M)
    return holding & separated
