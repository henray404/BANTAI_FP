"""Pure-tensor curriculum helpers: goal_id one-hot + goal-xyz anneal. No Isaac import."""

from __future__ import annotations

import torch


def goal_id_onehot(cat_idx: torch.Tensor, num_cats: int = 3) -> torch.Tensor:
    """(N,num_cats) float one-hot of the commanded category index (0=fragile,1=regular,2=heavy)."""
    return torch.nn.functional.one_hot(cat_idx.long(), num_classes=num_cats).float()


def anneal_goal(goal_xyz: torch.Tensor, alpha: float) -> torch.Tensor:
    """Scale goal xyz by alpha (1.0 = full leak, 0.0 = hidden). box_pos is annealed elsewhere (never)."""
    return goal_xyz * alpha
