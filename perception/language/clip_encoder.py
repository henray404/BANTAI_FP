# perception/language/clip_encoder.py
# Person 4 — frozen CLIP (ViT-B/32) text encoder for goal instructions.
#
# Only 3 distinct instructions exist (one per zone), so embeddings are computed
# ONCE and cached. The env then maps a per-env zone index → cached 512-d embedding
# with zero per-step CLIP cost.
#
# Backend: open_clip (pip: open-clip-torch). Import-guarded — if CLIP isn't
# installed the encoder reports unavailable and callers fall back to zeros, so the
# env never crashes (interface contract: goal_emb may be zeros until P4 is wired).

"""Frozen CLIP ViT-B/32 text encoder with 3-instruction caching."""

from __future__ import annotations

from .instructions import ALL_INSTRUCTIONS, ZONE_INSTRUCTIONS

GOAL_EMB_DIM = 512  # CLIP ViT-B/32 text projection dim — matches the obs contract

try:
    import open_clip  # type: ignore
    import torch

    _HAS_CLIP = True
except ImportError:
    open_clip = None
    _HAS_CLIP = False


class CLIPInstructionEncoder:
    """Encode the 3 zone instructions to frozen CLIP text embeddings, cached.

    embeddings() returns a (3, 512) tensor in ZONE_INSTRUCTIONS order, L2-normalized.
    available is False when open_clip is missing → callers use zeros.
    """

    def __init__(self, model_name: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k", device: str = "cpu"):
        """Lazy-load CLIP and precompute the 3 instruction embeddings."""
        self.available = _HAS_CLIP
        self.device = device
        self._emb = None
        if not self.available:
            return
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        tokenizer = open_clip.get_tokenizer(model_name)
        with torch.no_grad():
            tokens = tokenizer(ALL_INSTRUCTIONS).to(device)
            emb = model.encode_text(tokens).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
        self._emb = emb  # (3, 512)
        self._model = model

    def embeddings(self):
        """Return cached (3, 512) instruction embeddings (ZONE order)."""
        if not self.available:
            raise RuntimeError("CLIP unavailable — install open-clip-torch.")
        return self._emb

    def embed_zone_indices(self, zone_idx):
        """Map a tensor of zone indices (N,) → (N, 512) cached embeddings."""
        return self._emb[zone_idx]


def zone_index_from_goal_pos(goal_pos, zone_xyz):
    """Match each env's goal_pos (N,3) to the nearest zone → index (N,).

    `zone_xyz` is the (3,3) env-local zone centers (from ZONE_SPECS). Nearest by xy
    so a curriculum-annealed goal still resolves to the intended zone.
    """
    import torch

    d = torch.cdist(goal_pos[:, :2], zone_xyz[:, :2])  # (N, 3)
    return d.argmin(dim=-1)


# Number of zones, exported so callers can validate without importing scene.
NUM_ZONES = len(ZONE_INSTRUCTIONS)
