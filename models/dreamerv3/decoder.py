"""DreamerV3 decoder heads: image reconstruction, reward, continuation.

Architecture follows NM512 vendor patterns:
  - ConvDecoder: feat → linear → reshape → transposed convs → (B,3,64,64)
  - RewardHead: feat → MLP → scalar (MSE with symlog transform)
  - ContHead: feat → MLP → Bernoulli logit (binary cross-entropy)
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class ConvDecoder(nn.Module):
    def __init__(self, feat_size: int, shape: tuple = (3, 64, 64),
                 depth: int = 32, kernel: int = 4, minres: int = 4):
        super().__init__()
        self._shape = shape
        self._minres = minres
        ch, h, _ = shape
        n_layers = int(math.log2(h) - math.log2(minres))
        init_ch = depth * 2 ** (n_layers - 1)
        embed_size = init_ch * minres * minres

        self.linear = nn.Linear(feat_size, embed_size)
        self._embed_ch = init_ch

        layers = []
        in_ch = init_ch
        for i in range(n_layers):
            last = (i == n_layers - 1)
            out_ch = ch if last else in_ch // 2
            pad = kernel // 2 - 1
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel, stride=2,
                                             padding=pad, bias=last))
            if not last:
                layers.append(_ImgLayerNorm(out_ch))
                layers.append(nn.SiLU())
            in_ch = out_ch
        self.layers = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.linear(feat)
        x = x.reshape(-1, self._embed_ch, self._minres, self._minres)
        x = self.layers(x)
        return x + 0.5  # center around 0.5 like NM512


class _ImgLayerNorm(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.LayerNorm(ch, eps=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class RewardHead(nn.Module):
    def __init__(self, feat_size: int, hidden: int = 512, layers: int = 3):
        super().__init__()
        mods = []
        in_dim = feat_size
        for _ in range(layers):
            mods.extend([nn.Linear(in_dim, hidden, bias=False),
                         nn.LayerNorm(hidden, eps=1e-3), nn.SiLU()])
            in_dim = hidden
        mods.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)

    def loss(self, feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self.forward(feat)
        return F.mse_loss(pred, symlog(target), reduction="none")


class ContHead(nn.Module):
    def __init__(self, feat_size: int, hidden: int = 512, layers: int = 3):
        super().__init__()
        mods = []
        in_dim = feat_size
        for _ in range(layers):
            mods.extend([nn.Linear(in_dim, hidden, bias=False),
                         nn.LayerNorm(hidden, eps=1e-3), nn.SiLU()])
            in_dim = hidden
        mods.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)

    def loss(self, feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logit = self.forward(feat)
        return F.binary_cross_entropy_with_logits(logit, target, reduction="none")
