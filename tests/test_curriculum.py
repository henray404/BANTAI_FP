import torch
from env.curriculum import goal_id_onehot, anneal_goal


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
