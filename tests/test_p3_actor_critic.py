# tests/test_p3_actor_critic.py — pure-CPU unit tests (no Isaac, no GPU).
#   pytest tests/test_p3_actor_critic.py -v
"""Unit tests for policy.actor_critic — Actor, Critic, and lambda_return."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy.actor_critic import Actor, Critic, lambda_return

FEAT_DIM = 64    # small dim for fast CPU tests (prod = 1536)
ACTION_DIM = 6
B = 4
H = 8
DEVICE = "cpu"


# ── Actor ─────────────────────────────────────────────────────────────────────

def test_actor_output_shape():
    actor = Actor(feat_dim=FEAT_DIM, action_dim=ACTION_DIM, hidden=[32, 32])
    feat = torch.randn(B, FEAT_DIM)
    action, log_prob = actor(feat)
    assert action.shape == (B, ACTION_DIM)
    assert log_prob.shape == (B,)


def test_actor_action_in_bounds():
    actor = Actor(feat_dim=FEAT_DIM, action_dim=ACTION_DIM, hidden=[32, 32])
    feat = torch.randn(100, FEAT_DIM)
    action, _ = actor(feat)
    assert action.min().item() >= -1.0 - 1e-5
    assert action.max().item() <=  1.0 + 1e-5


def test_actor_mean_action():
    actor = Actor(feat_dim=FEAT_DIM, action_dim=ACTION_DIM, hidden=[32, 32])
    feat = torch.randn(B, FEAT_DIM)
    mean_a = actor.mean_action(feat)
    assert mean_a.shape == (B, ACTION_DIM)
    assert mean_a.min().item() >= -1.0 - 1e-5
    assert mean_a.max().item() <=  1.0 + 1e-5


def test_actor_entropy_positive():
    actor = Actor(feat_dim=FEAT_DIM, action_dim=ACTION_DIM, hidden=[32, 32])
    feat = torch.randn(B, FEAT_DIM)
    ent = actor.entropy(feat)
    assert ent.shape == (B,)
    assert (ent > 0).all()


def test_actor_loss_scalar_and_backward():
    actor = Actor(feat_dim=FEAT_DIM, action_dim=ACTION_DIM, hidden=[32, 32])
    feats = torch.randn(H, B, FEAT_DIM)
    returns = torch.randn(H, B)
    loss = actor.loss(feats, returns)
    assert loss.shape == ()
    loss.backward()  # should not raise


# ── Critic ────────────────────────────────────────────────────────────────────

def test_critic_output_shape():
    critic = Critic(feat_dim=FEAT_DIM, hidden=[32, 32])
    val = critic(torch.randn(B, FEAT_DIM))
    assert val.shape == (B, 1)


def test_critic_slow_value_no_grad():
    critic = Critic(feat_dim=FEAT_DIM, hidden=[32, 32])
    sv = critic.slow_value(torch.randn(B, FEAT_DIM))
    assert sv.shape == (B, 1)
    assert sv.grad_fn is None


def test_critic_loss_scalar():
    critic = Critic(feat_dim=FEAT_DIM, hidden=[32, 32])
    feats = torch.randn(H, B, FEAT_DIM)
    returns = torch.randn(H, B)
    loss = critic.loss(feats, returns)
    assert loss.shape == ()


def test_slow_target_updates():
    critic = Critic(feat_dim=FEAT_DIM, hidden=[32, 32], slow_target_freq=1)
    params_before = [p.data.clone() for p in critic.slow_net.parameters()]
    for p in critic.net.parameters():
        p.data.fill_(99.0)
    critic.update_slow_target()
    params_after = [p.data.clone() for p in critic.slow_net.parameters()]
    changed = any(not torch.allclose(b, a) for b, a in zip(params_before, params_after))
    assert changed, "Slow target must update after update_slow_target()"


# ── lambda_return ─────────────────────────────────────────────────────────────

def test_lambda_return_shape():
    rewards = torch.randn(H, B)
    values  = torch.randn(H + 1, B)
    dones   = torch.ones(H, B)
    ret = lambda_return(rewards, values, dones)
    assert ret.shape == (H, B)


def test_lambda_return_zero_dones_equals_rewards():
    """dones=0 everywhere (always terminal) → returns equal rewards."""
    rewards = torch.ones(H, B)
    values  = torch.randn(H + 1, B)
    dones   = torch.zeros(H, B)
    ret = lambda_return(rewards, values, dones, gamma=0.99, lambda_=0.95)
    torch.testing.assert_close(ret, rewards)


def test_lambda_return_td0():
    """lambda_=0 (TD-0) and values=0 → returns equal rewards."""
    H2, B2 = 3, 2
    rewards = torch.tensor([[1.0, 2.0]] * H2)
    values  = torch.zeros(H2 + 1, B2)
    dones   = torch.ones(H2, B2)
    ret = lambda_return(rewards, values, dones, gamma=1.0, lambda_=0.0)
    torch.testing.assert_close(ret, rewards)
