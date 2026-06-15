# perception/language/visual_her.py
# Person 4 — Visual HER relabeling (NOVELTY, Week 6).
#
# Standard HER relabels a failed episode by the POSITION the robot reached. Visual
# HER relabels by the CATEGORY the robot actually approached (what it saw): if the
# robot failed to reach zone A but got close to a heavy box, relabel the episode as
# a success for "deliver heavy". More semantically meaningful in multi-task setting.
#
# STATUS: design skeleton. The "which category did we approach" detector depends on
# P3's YOLO output (perception/detection) and/or box ground-truth proximity. Wire
# `approached_category()` once detection is online. relabel() shape matches
# training.replay_buffer.ReplayBuffer.her_relabel(relabel_fn=...).

"""Visual HER relabeling skeleton for the shared replay buffer."""

from __future__ import annotations

from typing import Callable

from .instructions import ZONE_INSTRUCTIONS

# category → (zone_index, zone_embedding-resolver). Embedding filled by the encoder.
_CATEGORY_TO_ZONE_IDX = {c: i for i, (_, c, _) in enumerate(ZONE_INSTRUCTIONS)}


def make_relabel_fn(
    encoder,
    zone_xyz,
    approached_category: Callable[[list[dict]], str | None],
    success_reward: float = 10.0,
) -> Callable[[list[dict]], list[dict]]:
    """Build a relabel_fn for ReplayBuffer.her_relabel.

    Args:
        encoder: CLIPInstructionEncoder (for the relabeled goal embedding).
        zone_xyz: (3,3) env-local zone centers — the relabeled goal position.
        approached_category: fn(trajectory) → category str the robot approached,
            or None if nothing meaningful was approached (then no relabel).
        success_reward: reward assigned to the achieved-goal transitions.

    Returns a fn(trajectory) → list[transition] suitable for her_relabel.
    """

    def relabel_fn(trajectory: list[dict]) -> list[dict]:
        cat = approached_category(trajectory)
        if cat is None or cat not in _CATEGORY_TO_ZONE_IDX:
            return []
        zidx = _CATEGORY_TO_ZONE_IDX[cat]
        new_goal_pos = zone_xyz[zidx]
        new_goal_emb = (
            encoder.embed_zone_indices(_as_idx_tensor(zidx))[0]
            if getattr(encoder, "available", False) else None
        )

        out: list[dict] = []
        n = len(trajectory)
        for i, t in enumerate(trajectory):
            obs = dict(t["obs"])
            next_obs = dict(t["next_obs"])
            obs["goal"] = _set_goal(obs.get("goal"), new_goal_pos)
            next_obs["goal"] = _set_goal(next_obs.get("goal"), new_goal_pos)
            if new_goal_emb is not None:
                obs["goal_emb"] = new_goal_emb
                next_obs["goal_emb"] = new_goal_emb
            # Achieved on the final transition of the approached-category episode.
            reward = success_reward if i == n - 1 else t["reward"]
            done = True if i == n - 1 else t["done"]
            out.append(dict(obs=obs, action=t["action"], reward=reward,
                            next_obs=next_obs, done=done))
        return out

    return relabel_fn


def _as_idx_tensor(zidx: int):
    """Wrap a scalar zone index as a (1,) long tensor for embed_zone_indices."""
    import torch

    return torch.tensor([zidx], dtype=torch.long)


def _set_goal(goal, new_goal_pos):
    """Replace a goal vector with the relabeled zone position (numpy/torch-safe)."""
    try:
        import torch

        if isinstance(new_goal_pos, torch.Tensor):
            return new_goal_pos.detach().cpu().numpy()
    except ImportError:
        pass
    return new_goal_pos
