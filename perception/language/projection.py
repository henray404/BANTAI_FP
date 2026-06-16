# perception/language/projection.py
# Person 4 — 512 → 64 projection for the goal embedding.
#
# CLIP output is 512-d; feeding that raw into the RSSM goal head is heavy. Project
# to 64-d (ref: LED-WM 2025 Sec.3, docs/research/referensi.md). This is a LEARNED layer that
# lives on the POLICY side (P2's RSSM), NOT in the env — the obs contract keeps
# goal_emb at 512 so the interface is unchanged. P2 instantiates this in the encoder.

"""Linear projection of the 512-d CLIP goal embedding to 64-d for the RSSM head."""

from __future__ import annotations

try:
    import torch
    from torch import nn

    _Base = nn.Module
except ImportError:
    _Base = object


class GoalProjection(_Base):
    """512 → 64 linear projection (+ optional LayerNorm). Trained with the policy."""

    def __init__(self, in_dim: int = 512, out_dim: int = 64, norm: bool = True):
        """Build the projection layer."""
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if norm else nn.Identity()

    def forward(self, x):
        """(B, 512) → (B, 64)."""
        return self.norm(self.linear(x))
