import torch
from env.grasp import grasp_success, grasp_lost, GRIP_RADIUS_M, LIFT_M


def _ee():
    return torch.tensor([[0.0, 0.0, 0.30]])


def _box():
    return torch.tensor([[0.0, 0.0, 0.30]])  # box at ee


def test_grasp_success_when_close_closed_and_lifted():
    ok = grasp_success(
        ee_pos=_ee(), box_pos=_box(),
        gripper_closed=torch.tensor([True]),
        box_lift=torch.tensor([LIFT_M + 0.01]),
    )
    assert bool(ok[0]) is True


def test_no_grasp_when_far():
    far_box = torch.tensor([[1.0, 0.0, 0.30]])
    ok = grasp_success(
        ee_pos=_ee(), box_pos=far_box,
        gripper_closed=torch.tensor([True]),
        box_lift=torch.tensor([LIFT_M + 0.01]),
    )
    assert bool(ok[0]) is False


def test_no_grasp_when_open():
    ok = grasp_success(
        ee_pos=_ee(), box_pos=_box(),
        gripper_closed=torch.tensor([False]),
        box_lift=torch.tensor([LIFT_M + 0.01]),
    )
    assert bool(ok[0]) is False


def test_grasp_lost_when_holding_and_ee_leaves_box():
    lost = grasp_lost(
        holding=torch.tensor([True]),
        ee_pos=_ee(), box_pos=torch.tensor([[0.5, 0.0, 0.30]]),
    )
    assert bool(lost[0]) is True
