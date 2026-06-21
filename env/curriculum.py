"""Pure-tensor curriculum helpers: goal_id one-hot + goal-xyz anneal + 4-stage manager logic.

No Isaac import — every function here is unit-testable (see tests/test_curriculum.py). The
sim-side wiring (per-stage reset, robot teleport, box pre-grasp) lives in
env.warehouse_env.WarehouseRLEnv and calls these helpers for the policy decisions.
"""

from __future__ import annotations

import math

import torch


# ── 4-stage curriculum (spec §7) ──────────────────────────────────────
# Stage isolates one skill so the policy learns from dense signal before facing the full chain.
STAGE_NAV    = 1  # box PRE-GRASPED at spawn → isolate carry + place
STAGE_GRASP  = 2  # spawn the base NEXT TO the target box → isolate approach + grasp
STAGE_FULL   = 3  # full chain nav→grasp→carry→place (default, = legacy behavior)
STAGE_ANNEAL = 4  # full chain but goal xyz annealed toward 0 → deliver from goal_id + pixels alone
VALID_STAGES = (STAGE_NAV, STAGE_GRASP, STAGE_FULL, STAGE_ANNEAL)


def goal_id_onehot(cat_idx: torch.Tensor, num_cats: int = 3) -> torch.Tensor:
    """(N,num_cats) float one-hot of the commanded category index (0=fragile,1=regular,2=heavy)."""
    return torch.nn.functional.one_hot(cat_idx.long(), num_classes=num_cats).float()


def anneal_goal(goal_xyz: torch.Tensor, alpha: float) -> torch.Tensor:
    """Scale goal xyz by alpha (1.0 = full leak, 0.0 = hidden). box_pos is annealed elsewhere (never)."""
    return goal_xyz * alpha


def validate_stage(stage: int) -> int:
    """Return stage as int if it is one of 1..4, else raise ValueError."""
    s = int(stage)
    if s not in VALID_STAGES:
        raise ValueError(f"stage must be one of {VALID_STAGES}, got {stage!r}")
    return s


def stage_is_pregrasped(stage: int) -> bool:
    """True if this stage spawns the target box already held (Stage 1 Nav-only)."""
    return int(stage) == STAGE_NAV


def stage_is_spawn_near_box(stage: int) -> bool:
    """True if this stage overrides the robot spawn to be next to the box (Stage 2 Grasp-only)."""
    return int(stage) == STAGE_GRASP


def resolve_goal_alpha(stage: int, anneal_alpha: float) -> float:
    """Goal-leak factor for the current stage.

    Stages 1..3 always leak the full goal xyz (alpha=1.0); only Stage 4 honors the externally
    scheduled `anneal_alpha` (driven down toward 0 by P3/P5). Clamped to [0,1].
    """
    if int(stage) == STAGE_ANNEAL:
        return float(min(1.0, max(0.0, anneal_alpha)))
    return 1.0


def spawn_pose_near_box(
    box_xy: tuple[float, float],
    standoff: float = 0.8,
    approach_dir: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, float, float]:
    """Base (x, y, yaw) placed `standoff` m from the box along `approach_dir`, facing the box.

    Stage-2 grasp isolation: drop the robot within Franka reach of the target box so the policy
    practises approach+grasp without first solving navigation. `approach_dir` is the unit vector
    FROM the box TO the robot (default +y = north, the open receiving side). yaw points the
    chassis back at the box so its onboard camera frames it.
    """
    bx, by = float(box_xy[0]), float(box_xy[1])
    dx, dy = float(approach_dir[0]), float(approach_dir[1])
    norm = math.hypot(dx, dy) or 1.0
    dx, dy = dx / norm, dy / norm
    base_x = bx + dx * standoff
    base_y = by + dy * standoff
    yaw = math.atan2(by - base_y, bx - base_x)  # face from base toward box
    return base_x, base_y, yaw
