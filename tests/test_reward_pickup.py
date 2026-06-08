import torch
from types import SimpleNamespace
from env.reward_pickup import (
    approach_box_distance, carry_distance, grasp_success_reward,
    drop_penalty, pickup_delivered, pickup_delivered_reward,
)


def _env(**kw):
    base = dict(
        num_envs=1, device="cpu",
        ee_pos=torch.tensor([[0.0, 0.0, 0.3]]),
        box_pos=torch.tensor([[0.0, 0.0, 0.3]]),
        holding=torch.tensor([False]),
        goal_pos=torch.tensor([[0.0, -12.0, 0.0]]),
        grasp_event=torch.tensor([False]),
        drop_event=torch.tensor([False]),
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_approach_distance_zero_when_ee_on_box():
    assert torch.allclose(approach_box_distance(_env()), torch.tensor([0.0]))


def test_approach_gated_off_when_holding():
    e = _env(holding=torch.tensor([True]), ee_pos=torch.tensor([[1.0, 0.0, 0.3]]))
    assert torch.allclose(approach_box_distance(e), torch.tensor([0.0]))  # gated -> 0


def test_carry_distance_active_only_when_holding():
    e = _env(holding=torch.tensor([True]),
             box_pos=torch.tensor([[0.0, 0.0, 0.3]]),
             goal_pos=torch.tensor([[0.0, -3.0, 0.0]]))
    assert torch.allclose(carry_distance(e), torch.tensor([3.0]))
    assert torch.allclose(carry_distance(_env()), torch.tensor([0.0]))  # not holding -> 0


def test_grasp_reward_fires_on_event():
    assert torch.allclose(grasp_success_reward(_env(grasp_event=torch.tensor([True]))),
                          torch.tensor([1.0]))


def test_drop_penalty_fires_on_event():
    assert torch.allclose(drop_penalty(_env(drop_event=torch.tensor([True]))),
                          torch.tensor([1.0]))


def test_pickup_delivered_requires_holding_and_in_zone():
    e = _env(holding=torch.tensor([True]),
             box_pos=torch.tensor([[0.0, -12.0, 0.3]]),
             goal_pos=torch.tensor([[0.0, -12.0, 0.0]]))
    assert bool(pickup_delivered(e)[0]) is True
    assert torch.allclose(pickup_delivered_reward(e), torch.tensor([1.0]))
    assert bool(pickup_delivered(_env())[0]) is False
