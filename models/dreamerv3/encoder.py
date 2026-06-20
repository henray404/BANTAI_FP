"""DreamerV3 encoder: CNN (pixels) + MLP (low-dim obs) → embedding.

Architecture follows NM512 vendor ConvEncoder/MultiEncoder patterns:
  - 4 stages of stride-2 Conv2d, depth doubling, LayerNorm + SiLU
  - 64×64 → 4×4 spatial, depth 32→256, flatten → 4096
  - Low-dim (19) → 2-layer MLP → 256
  - Concat → embed_dim = 4352
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


LOW_DIM_KEYS = (
    "position", "heading", "goal", "goal_id",
    "ee_pos", "gripper", "holding", "box_pos",
)
LOW_DIM_TOTAL = 3 + 2 + 3 + 3 + 3 + 1 + 1 + 3  # = 19


class Conv2dSamePad(nn.Conv2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]
        pad_h = max((math.ceil(ih / self.stride[0]) - 1) * self.stride[0]
                     + self.kernel_size[0] - ih, 0)
        pad_w = max((math.ceil(iw / self.stride[1]) - 1) * self.stride[1]
                     + self.kernel_size[1] - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2,
                          pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ImgLayerNorm(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.LayerNorm(ch, eps=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) → (B, H, W, C) → norm → (B, C, H, W)
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvEncoder(nn.Module):
    def __init__(self, depth: int = 32, kernel: int = 4, minres: int = 4,
                 in_ch: int = 3, img_size: int = 64):
        super().__init__()
        stages = int(math.log2(img_size) - math.log2(minres))
        layers = []
        out_dim = depth
        h = img_size
        for _ in range(stages):
            layers.extend([
                Conv2dSamePad(in_ch, out_dim, kernel, stride=2, bias=False),
                ImgLayerNorm(out_dim),
                nn.SiLU(),
            ])
            in_ch = out_dim
            out_dim *= 2
            h //= 2
        self.layers = nn.Sequential(*layers)
        self.out_dim = (out_dim // 2) * h * h

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        # pixels: (B, 3, 64, 64) float [0,1]
        x = pixels - 0.5
        x = self.layers(x)
        return x.reshape(x.shape[0], -1)


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int = LOW_DIM_TOTAL, hidden: int = 256, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden, eps=1e-3), nn.SiLU(),
            nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim, eps=1e-3), nn.SiLU(),
        )
        self.out_dim = out_dim

    def forward(self, low_dim: torch.Tensor) -> torch.Tensor:
        return self.net(low_dim)


class Encoder(nn.Module):
    def __init__(self, depth: int = 32, kernel: int = 4, minres: int = 4,
                 mlp_hidden: int = 256, mlp_out: int = 256):
        super().__init__()
        self.cnn = ConvEncoder(depth, kernel, minres)
        self.mlp = MLPEncoder(LOW_DIM_TOTAL, mlp_hidden, mlp_out)
        self.out_dim = self.cnn.out_dim + self.mlp.out_dim

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        cnn_out = self.cnn(obs["pixels"])
        parts = [obs[k] for k in LOW_DIM_KEYS]
        low_dim = torch.cat(parts, dim=-1)
        mlp_out = self.mlp(low_dim)
        return torch.cat([cnn_out, mlp_out], dim=-1)
