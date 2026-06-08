"""Pure-tensor split of the external (N,6) pickup action. No Isaac import (unit-testable)."""

from __future__ import annotations

import torch

EE_STEP_M = 0.05  # meters of EE travel commanded per control step at action == 1.0


def split_action(action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split (N,6) [base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper] into (base2, ee3, grip1).

    The EE delta is scaled from [-1,1] to a Cartesian step (EE_STEP_M); base and gripper pass through.
    """
    base = action[:, 0:2]
    ee = action[:, 2:5] * EE_STEP_M
    grip = action[:, 5:6]
    return base, ee, grip
