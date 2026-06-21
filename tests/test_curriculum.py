import math

import pytest
import torch

from env.curriculum import (
    STAGE_ANNEAL,
    STAGE_FULL,
    STAGE_GRASP,
    STAGE_NAV,
    anneal_goal,
    goal_id_onehot,
    resolve_goal_alpha,
    spawn_pose_near_box,
    stage_is_pregrasped,
    stage_is_spawn_near_box,
    validate_stage,
)


def test_goal_id_onehot():
    idx = torch.tensor([0, 2, 1])
    oh = goal_id_onehot(idx, num_cats=3)
    assert oh.shape == (3, 3)
    assert torch.allclose(oh[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(oh[1], torch.tensor([0.0, 0.0, 1.0]))


def test_anneal_full_then_zero():
    goal = torch.tensor([[3.0, -12.0, 0.0]])
    assert torch.allclose(anneal_goal(goal, alpha=1.0), goal)
    assert torch.allclose(anneal_goal(goal, alpha=0.0), torch.zeros_like(goal))
    assert torch.allclose(anneal_goal(goal, alpha=0.5), goal * 0.5)


# ── stage manager logic ───────────────────────────────────────────────
def test_validate_stage_accepts_1_to_4():
    for s in (1, 2, 3, 4):
        assert validate_stage(s) == s


def test_validate_stage_rejects_out_of_range():
    for bad in (0, 5, -1):
        with pytest.raises(ValueError):
            validate_stage(bad)


def test_stage_flags():
    assert stage_is_pregrasped(STAGE_NAV) is True
    assert stage_is_pregrasped(STAGE_GRASP) is False
    assert stage_is_spawn_near_box(STAGE_GRASP) is True
    assert stage_is_spawn_near_box(STAGE_FULL) is False


def test_resolve_goal_alpha_only_anneals_stage4():
    # Stages 1-3 always leak the full goal regardless of the scheduled alpha.
    for s in (STAGE_NAV, STAGE_GRASP, STAGE_FULL):
        assert resolve_goal_alpha(s, anneal_alpha=0.0) == 1.0
    # Stage 4 honors the scheduled alpha, clamped to [0,1].
    assert resolve_goal_alpha(STAGE_ANNEAL, 0.3) == pytest.approx(0.3)
    assert resolve_goal_alpha(STAGE_ANNEAL, 0.0) == 0.0
    assert resolve_goal_alpha(STAGE_ANNEAL, 2.0) == 1.0
    assert resolve_goal_alpha(STAGE_ANNEAL, -1.0) == 0.0


def test_spawn_pose_near_box_default_north():
    # Box at origin, default approach from +y (north): base sits north of box, faces south (-y).
    bx, by, yaw = spawn_pose_near_box((0.0, 0.0), standoff=0.8)
    assert bx == pytest.approx(0.0)
    assert by == pytest.approx(0.8)
    assert yaw == pytest.approx(-math.pi / 2)  # facing -y toward the box


def test_spawn_pose_near_box_standoff_distance():
    # Base is exactly `standoff` from the box along any approach direction.
    box = (3.0, -5.0)
    bx, by, _ = spawn_pose_near_box(box, standoff=0.7, approach_dir=(1.0, 0.0))
    assert math.hypot(bx - box[0], by - box[1]) == pytest.approx(0.7)
    assert bx == pytest.approx(3.7)
    assert by == pytest.approx(-5.0)
