import torch
from env.grasp import grasp_success, grasp_lost, GRIP_RADIUS_M

_BOX_HALF = torch.tensor([0.16])  # regular box edge 0.32 / 2


def _ee():
    return torch.tensor([[0.0, 0.0, 0.30]])


def _box():
    return torch.tensor([[0.0, 0.0, 0.30]])  # box center at ee


def test_grasp_success_when_close_and_closed():
    ok = grasp_success(
        ee_pos=_ee(), box_pos=_box(),
        gripper_closed=torch.tensor([True]),
        box_half=_BOX_HALF,
    )
    assert bool(ok[0]) is True


def test_grasp_success_at_surface_of_large_box():
    """EE far from CENTER but within GRIP_RADIUS_M of the SURFACE still grasps (oversized box)."""
    heavy_half = torch.tensor([0.26])               # heavy box edge 0.52 / 2
    ee = torch.tensor([[0.30, 0.0, 0.30]])          # 0.30 m from center > old 0.08 radius
    box = torch.tensor([[0.0, 0.0, 0.30]])          # surface_dist = 0.30 - 0.26 = 0.04 < 0.10
    ok = grasp_success(ee, box, torch.tensor([True]), heavy_half)
    assert bool(ok[0]) is True


def test_no_grasp_when_far():
    far_box = torch.tensor([[1.0, 0.0, 0.30]])      # surface_dist = 1.0 - 0.16 >> 0.10
    ok = grasp_success(
        ee_pos=_ee(), box_pos=far_box,
        gripper_closed=torch.tensor([True]),
        box_half=_BOX_HALF,
    )
    assert bool(ok[0]) is False


def test_no_grasp_when_open():
    ok = grasp_success(
        ee_pos=_ee(), box_pos=_box(),
        gripper_closed=torch.tensor([False]),
        box_half=_BOX_HALF,
    )
    assert bool(ok[0]) is False


def test_grasp_lost_when_holding_and_ee_leaves_box():
    lost = grasp_lost(
        holding=torch.tensor([True]),
        ee_pos=_ee(), box_pos=torch.tensor([[0.5, 0.0, 0.30]]),
        box_half=_BOX_HALF,
    )
    assert bool(lost[0]) is True
