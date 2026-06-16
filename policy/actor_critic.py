# policy/actor_critic.py
# P3 (Jeremy) — Actor + Critic for DreamerV3 imagination-based training.
#
# Both networks operate on the RSSM feature vector produced by P2's world model:
#   feat = cat(rssm_deter, rssm_stoch.flatten())   shape: (B, feat_dim)
#   default feat_dim = 512 + 32*32 = 1536 (DreamerV3-small configs.yaml defaults)
#
# Actor: feat → action (B, 6) in [-1, 1]   action = [base_lin, base_ang, ee_dx/dy/dz, gripper]
# Critic: feat → value (B, 1) scalar
# lambda_return(): GAE-λ returns over imagined trajectory
#
# No Isaac Lab import here — pure PyTorch, unit-testable without the simulator.

"""Actor, Critic, and lambda-return for DreamerV3 imagination-based RL."""

from __future__ import annotations

import copy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Utilities ─────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden: list[int], out_dim: int, act=nn.SiLU) -> nn.Sequential:
    """MLP with LayerNorm after each hidden layer."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), act()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ── Actor ─────────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """Gaussian actor conditioned on the DreamerV3 RSSM feature.

    Outputs a tanh-squashed action sample + log-probability for the policy
    gradient update. Training objective: maximize lambda-returns + entropy bonus.

    Args:
        feat_dim:      RSSM feature dimension (default 1536 for DreamerV3-small).
        action_dim:    Action dimensions (default 6 per P1 new contract).
        hidden:        Hidden layer sizes.
        min_std:       Minimum std (exploration floor).
        max_std:       Maximum std (exploration ceiling).
        entropy_scale: Weight for entropy bonus in loss.
    """

    def __init__(
        self,
        feat_dim: int = 1536,
        action_dim: int = 6,
        hidden: list[int] | None = None,
        min_std: float = 0.1,
        max_std: float = 1.0,
        entropy_scale: float = 3e-4,
    ):
        super().__init__()
        hidden = hidden or [512, 256]
        self.action_dim = action_dim
        self.min_std = min_std
        self.max_std = max_std
        self.entropy_scale = entropy_scale
        # Two outputs: mean (action_dim) + log_std (action_dim)
        self.net = _mlp(feat_dim, hidden, action_dim * 2)

    def _get_dist(self, feat: Tensor) -> torch.distributions.Distribution:
        """Build a TanhNormal distribution from the RSSM feature."""
        raw = self.net(feat)
        mean, log_std = raw.chunk(2, dim=-1)
        std = torch.sigmoid(log_std) * (self.max_std - self.min_std) + self.min_std
        base = torch.distributions.Normal(mean, std)
        return torch.distributions.TransformedDistribution(
            base,
            torch.distributions.transforms.TanhTransform(cache_size=1),
        )

    def forward(self, feat: Tensor) -> Tuple[Tensor, Tensor]:
        """Sample a tanh-squashed action and compute its log-probability.

        Args:
            feat: (*, feat_dim) RSSM features.

        Returns:
            action:   (*, action_dim) in [-1, 1].
            log_prob: (*,) summed log-probability.
        """
        dist = self._get_dist(feat)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    def mean_action(self, feat: Tensor) -> Tensor:
        """Deterministic mean action for evaluation (no sampling noise)."""
        raw = self.net(feat)
        mean, _ = raw.chunk(2, dim=-1)
        return torch.tanh(mean)

    def entropy(self, feat: Tensor) -> Tensor:
        """Entropy of the policy distribution. Shape: (*,)."""
        dist = self._get_dist(feat)
        # TanhTransform doesn't implement entropy analytically; use base Normal entropy.
        # This is an approximation — exact entropy would require log-det of tanh Jacobian.
        return dist.base_dist.entropy().sum(-1)

    def loss(self, imagine_feat: Tensor, lambda_returns: Tensor) -> Tensor:
        """Actor loss = -E[lambda_returns * log_prob] - entropy_scale * E[entropy].

        Args:
            imagine_feat:   (H, B, feat_dim) imagined RSSM features.
            lambda_returns: (H, B) lambda-return targets (detached from critic graph).

        Returns:
            Scalar loss tensor.
        """
        flat_feat = imagine_feat.flatten(0, 1)           # (H*B, feat_dim)
        _, log_prob = self.forward(flat_feat)
        log_prob = log_prob.view(imagine_feat.shape[:2])  # (H, B)
        ent = self.entropy(flat_feat).view(imagine_feat.shape[:2])

        returns_norm = lambda_returns.detach()
        policy_loss = -(returns_norm * log_prob).mean()
        entropy_loss = -self.entropy_scale * ent.mean()
        return policy_loss + entropy_loss


# ── Critic ─────────────────────────────────────────────────────────────────────

class Critic(nn.Module):
    """Scalar value critic conditioned on the DreamerV3 RSSM feature.

    Includes a slow-target (EMA copy) for stable lambda-return bootstrap targets,
    matching DreamerV3's slow-critic design.

    Args:
        feat_dim:         RSSM feature dimension.
        hidden:           Hidden layer sizes.
        slow_target_freq: Steps between slow-target EMA updates.
        slow_frac:        EMA coefficient.
    """

    def __init__(
        self,
        feat_dim: int = 1536,
        hidden: list[int] | None = None,
        slow_target_freq: int = 100,
        slow_frac: float = 0.02,
    ):
        super().__init__()
        hidden = hidden or [512, 256]
        self.slow_target_freq = slow_target_freq
        self.slow_frac = slow_frac
        self._step = 0

        self.net = _mlp(feat_dim, hidden, 1)
        self.slow_net = copy.deepcopy(self.net)
        for p in self.slow_net.parameters():
            p.requires_grad_(False)

    def forward(self, feat: Tensor) -> Tensor:
        """Return predicted value. Shape: (*, 1)."""
        return self.net(feat)

    def slow_value(self, feat: Tensor) -> Tensor:
        """Slow-target value (no grad). Shape: (*, 1)."""
        with torch.no_grad():
            return self.slow_net(feat)

    def update_slow_target(self) -> None:
        """Call once per update step; EMA-syncs slow target every slow_target_freq steps."""
        self._step += 1
        if self._step % self.slow_target_freq != 0:
            return
        for main_p, slow_p in zip(self.net.parameters(), self.slow_net.parameters()):
            slow_p.data.lerp_(main_p.data, self.slow_frac)

    def loss(self, imagine_feat: Tensor, lambda_returns: Tensor) -> Tensor:
        """Critic loss = MSE(V(feat), lambda_returns.detach()).

        Args:
            imagine_feat:   (H, B, feat_dim) imagined features.
            lambda_returns: (H, B) lambda-return targets (detached).

        Returns:
            Scalar loss tensor.
        """
        flat = imagine_feat.flatten(0, 1)                    # (H*B, feat_dim)
        values = self.forward(flat).squeeze(-1)              # (H*B,)
        values = values.view(imagine_feat.shape[:2])         # (H, B)
        return F.mse_loss(values, lambda_returns.detach())


# ── Lambda-return ──────────────────────────────────────────────────────────────

def lambda_return(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    gamma: float = 0.997,
    lambda_: float = 0.95,
) -> Tensor:
    """Compute TD(λ) lambda-returns over an imagined trajectory.

    V_λ(t) = r(t) + γ*(1-d(t)) * [(1-λ)*V(t+1) + λ*V_λ(t+1)]
    Boundary: V_λ(H) = V(H)  (bootstrap from last critic value)

    Args:
        rewards: (H, B) imagined rewards.
        values:  (H+1, B) critic values; last entry is bootstrap V(H).
        dones:   (H, B) float continuation (1=not done, 0=done).
        gamma:   Discount factor.
        lambda_: TD(λ) mixing coefficient.

    Returns:
        returns: (H, B) lambda-return targets.
    """
    H = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    next_ret = values[H]

    for t in reversed(range(H)):
        td = rewards[t] + gamma * dones[t] * values[t + 1]
        next_ret = td + gamma * lambda_ * dones[t] * (next_ret - values[t + 1])
        returns[t] = next_ret

    return returns
