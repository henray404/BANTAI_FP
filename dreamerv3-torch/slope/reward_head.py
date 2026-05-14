# slope/reward_head.py
# SLOPE Reward Head — Quantile network + potential function + QCE loss
# Ref: Li et al., 2026 (arxiv.org/abs/2602.03201)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SLOPERewardHead(nn.Module):
    """
    SLOPE Reward Head untuk DreamerV3.

    Menggantikan reward head standar (MSE) dengan:
    - Quantile network: prediksi distribusi reward via N quantiles
    - Potential function: shaped reward = E[Q(s')] - E[Q(s)]
    - QCE loss: Quantile Cross-Entropy sebagai pengganti MSE

    Args:
        input_dim  : dimensi latent state dari DreamerV3 (default 1024)
        hidden_dim : ukuran hidden layer (default 512)
        num_quantiles: jumlah quantile (default 32, sesuai paper)
        gamma      : discount factor untuk potential shaping (default 0.99)
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_quantiles: int = 32,
        gamma: float = 0.99,
    ):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.gamma = gamma

        # Quantile network: latent state -> N quantile values
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_quantiles),
        )

        # Fixed quantile midpoints: tau_i = (2i - 1) / (2N)
        # Register as buffer supaya ikut .to(device) otomatis
        taus = (2 * torch.arange(1, num_quantiles + 1) - 1) / (2 * num_quantiles)
        self.register_buffer("taus", taus)  # shape: (num_quantiles,)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Predict quantile values dari latent state.

        Args:
            latent: tensor shape (..., input_dim)
        Returns:
            quantiles: tensor shape (..., num_quantiles)
        """
        return self.network(latent)

    # ------------------------------------------------------------------
    # Potential function — untuk shaped reward
    # ------------------------------------------------------------------

    def potential(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Potential Phi(s) = mean across quantiles.
        Scalar summary dari distribusi reward di state s.

        Args:
            latent: (..., input_dim)
        Returns:
            phi: (...,) — scalar potential per state
        """
        quantiles = self.forward(latent)      # (..., num_quantiles)
        return quantiles.mean(dim=-1)         # (...,)

    def shaped_reward(
        self,
        reward: torch.Tensor,
        latent_curr: torch.Tensor,
        latent_next: torch.Tensor,
    ) -> torch.Tensor:
        """
        Shaped reward = r + gamma * Phi(s') - Phi(s)
        (potential-based reward shaping, policy-invariant)

        Args:
            reward      : (...,) — reward asli dari environment
            latent_curr : (..., input_dim) — latent state saat ini
            latent_next : (..., input_dim) — latent state berikutnya
        Returns:
            r_shaped: (...,) — reward setelah di-shape
        """
        phi_curr = self.potential(latent_curr)   # (...,)
        phi_next = self.potential(latent_next)   # (...,)
        return reward + self.gamma * phi_next - phi_curr

    # ------------------------------------------------------------------
    # QCE Loss — Quantile Cross-Entropy
    # ------------------------------------------------------------------

    def qce_loss(
        self,
        latent: torch.Tensor,
        target_reward: torch.Tensor,
    ) -> torch.Tensor:
        """
        Quantile Cross-Entropy loss (QCE).

        Untuk setiap quantile tau_i, QCE loss adalah asymmetric L1:
          L_i = (tau_i - 1(target < q_i)) * (target - q_i)

        Total loss = mean over semua quantiles dan samples.

        Args:
            latent        : (..., input_dim)
            target_reward : (...,) — reward target (dari environment)
        Returns:
            loss: scalar tensor
        """
        quantiles = self.forward(latent)   # (..., num_quantiles)

        # Expand untuk broadcasting: (..., 1) vs (..., num_quantiles)
        target = target_reward.unsqueeze(-1)          # (..., 1)
        error = target - quantiles                    # (..., num_quantiles)

        # Asymmetric weight berdasarkan tanda error
        # taus shape: (num_quantiles,) — broadcast otomatis
        weight = torch.abs(
            self.taus - (error.detach() < 0).float()
        )                                             # (..., num_quantiles)

        loss = (weight * F.huber_loss(
            quantiles, target.expand_as(quantiles),
            reduction="none", delta=1.0
        )).mean()

        return loss

    # ------------------------------------------------------------------
    # Convenience: combined loss untuk training loop
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        latent: torch.Tensor,
        target_reward: torch.Tensor,
    ) -> dict:
        """
        Return dict berisi loss dan potential values untuk W&B logging.

        Returns:
            {
              'loss'     : scalar — QCE loss untuk backprop,
              'potential': scalar — mean potential (untuk W&B monitoring),
              'quantile_mean': scalar — mean of predicted quantiles,
            }
        """
        loss = self.qce_loss(latent, target_reward)
        with torch.no_grad():
            phi = self.potential(latent).mean()
            q_mean = self.forward(latent).mean()

        return {
            "loss": loss,
            "potential": phi,
            "quantile_mean": q_mean,
        }
