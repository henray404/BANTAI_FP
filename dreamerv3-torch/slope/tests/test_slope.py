# slope/tests/test_slope.py
# Unit tests untuk SLOPERewardHead

import pytest
import torch
from slope.reward_head import SLOPERewardHead


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def head():
    return SLOPERewardHead(input_dim=64, hidden_dim=128, num_quantiles=16)

@pytest.fixture
def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@pytest.fixture
def dummy(device):
    """Batch of 8 latent states, dim=64"""
    return torch.randn(8, 64, device=device)


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #

def test_forward_shape(head, dummy, device):
    """Output quantiles harus shape (batch, num_quantiles)"""
    head = head.to(device)
    out = head(dummy)
    assert out.shape == (8, 16), f"Expected (8,16), got {out.shape}"

def test_potential_shape(head, dummy, device):
    """Potential harus scalar per state: shape (batch,)"""
    head = head.to(device)
    phi = head.potential(dummy)
    assert phi.shape == (8,), f"Expected (8,), got {phi.shape}"

def test_shaped_reward_shape(head, dummy, device):
    """Shaped reward harus same shape sebagai reward input"""
    head = head.to(device)
    reward = torch.zeros(8, device=device)
    latent_next = torch.randn(8, 64, device=device)
    r_shaped = head.shaped_reward(reward, dummy, latent_next)
    assert r_shaped.shape == (8,), f"Expected (8,), got {r_shaped.shape}"

def test_qce_loss_scalar(head, dummy, device):
    """QCE loss harus scalar dan tidak NaN"""
    head = head.to(device)
    target = torch.randn(8, device=device)
    loss = head.qce_loss(dummy, target)
    assert loss.shape == (), f"Loss harus scalar, got {loss.shape}"
    assert not torch.isnan(loss), "Loss NaN!"
    assert not torch.isinf(loss), "Loss Inf!"

def test_loss_decreases(head, device):
    """Loss harus turun setelah beberapa gradient steps"""
    head = head.to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    latent = torch.randn(32, 64, device=device)
    target = torch.randn(32, device=device)

    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        result = head.compute_loss(latent, target)
        result["loss"].backward()
        optimizer.step()
        losses.append(result["loss"].item())

    assert losses[-1] < losses[0], (
        f"Loss tidak turun: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )

def test_compute_loss_keys(head, dummy, device):
    """compute_loss harus return dict dengan keys yang benar"""
    head = head.to(device)
    target = torch.randn(8, device=device)
    result = head.compute_loss(dummy, target)
    assert "loss" in result
    assert "potential" in result
    assert "quantile_mean" in result

def test_no_nan_on_zero_reward(head, device):
    """Tidak boleh NaN meski reward semua nol (edge case sparse reward)"""
    head = head.to(device)
    latent = torch.randn(16, 64, device=device)
    target = torch.zeros(16, device=device)   # sparse reward scenario
    result = head.compute_loss(latent, target)
    assert not torch.isnan(result["loss"]), "NaN pada zero reward!"

def test_taus_buffer_on_device(head, device):
    """taus buffer harus ikut pindah ke device yang sama"""
    head = head.to(device)
    assert head.taus.device.type == device.type
