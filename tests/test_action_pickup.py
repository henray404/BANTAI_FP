import torch
from env.action_pickup import split_action, EE_STEP_M


def test_split_shapes():
    a = torch.zeros(4, 6)
    base, ee, grip = split_action(a)
    assert base.shape == (4, 2)
    assert ee.shape == (4, 3)
    assert grip.shape == (4, 1)


def test_ee_delta_scaled_by_step():
    a = torch.zeros(1, 6)
    a[0, 2] = 1.0  # full +x ee command
    _, ee, _ = split_action(a)
    assert torch.allclose(ee[0], torch.tensor([EE_STEP_M, 0.0, 0.0]))


def test_passthrough_base_and_gripper():
    a = torch.tensor([[0.3, -0.7, 0.0, 0.0, 0.0, 0.9]])
    base, _, grip = split_action(a)
    assert torch.allclose(base[0], torch.tensor([0.3, -0.7]))
    assert torch.allclose(grip[0], torch.tensor([0.9]))
