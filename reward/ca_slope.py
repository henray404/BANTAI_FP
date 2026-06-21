# reward/ca_slope.py
# Person 5 — Category-Aware SLOPE (CA-SLOPE): potential-based dense reward shaping (RQ2).
#
# Potential-based reward shaping (Ng, Harada & Russell 1999): adding
#     F(s, s') = gamma * Phi(s') - Phi(s)
# to the base reward leaves the set of optimal policies UNCHANGED for ANY potential Phi.
# CA-SLOPE makes Phi *category-aware*: the shaping gain is read per-category from the one-hot
# `goal_id` (NOT from vision — YOLO removed 2026-06-08). The RQ2 ablation is per-category gains
# (category_aware=True) vs a single generic gain (category_aware=False) — same code path, one flag.
#
# BACKEND-AGNOSTIC: every op here (slice, *, +, sum(-1), **0.5) is valid for BOTH numpy arrays and
# torch tensors. So the SAME module computes shaping inside the torch training reward (on the Isaac
# box) AND inside the numpy headless eval harness (on a Mac, no Isaac). There is no torch/numpy
# import — it operates on whatever array type you pass in.

"""Category-Aware SLOPE — potential-based, category-conditioned dense reward shaping (P5)."""

from __future__ import annotations

# Category index order MUST match the goal_id one-hot (env/curriculum.goal_id_onehot):
#   0 = fragile (orange, 21cm) · 1 = regular (cyan, 32cm) · 2 = heavy (purple, 52cm)
CATEGORY_NAMES: tuple[str, str, str] = ("fragile", "regular", "heavy")

DEFAULT_GAMMA: float = 0.997  # match the agent discount; PBRS invariance is exact only at this gamma

# Per-category shaping gain. Heavier boxes get a steeper landscape (more dense guidance) because
# they are the hardest to learn; fragile gets the gentlest. This per-category spread is the whole
# point of CA-SLOPE — generic SLOPE collapses all three to DEFAULT_GENERIC_GAIN (the RQ2 control).
DEFAULT_CATEGORY_GAINS: tuple[float, float, float] = (1.0, 1.5, 2.0)
DEFAULT_GENERIC_GAIN: float = 1.5

# Phase-A potential carries a constant offset ~= the nominal box->zone carry distance so the
# potential is (near-)continuous when `holding` flips 0->1 at grasp. Without it, Phi would jump
# from ~0 (ee on box) to -gain*carry_dist (just grasped, box far from zone), and F would punish the
# grasp it should reward. Any offset preserves PBRS invariance; this one just makes Phi monotone
# from spawn to delivery. Default ~= receiving-row (y=+1) to delivery-zone (y=-12) distance.
DEFAULT_PHASE_B_OFFSET: float = 13.0


def _norm_last(x):
    """Euclidean norm over the last axis. (x*x).sum(-1)**0.5 is valid for numpy AND torch."""
    return ((x * x).sum(-1)) ** 0.5


def _squeeze_flag(h):
    """`holding` is (N,1) in the OBS dict but (N,) in the env buffer. Collapse a trailing 1 only.

    Guarded on ndim>=2 so a genuine (N,) flag (incl. the common num_envs=1 -> (1,)) is left alone.
    """
    if getattr(h, "ndim", 1) >= 2 and h.shape[-1] == 1:
        return h[..., 0]
    return h


def state_from_obs(obs: dict, goal_key: str = "goal") -> dict:
    """Map a v2 obs dict (or buffer Batch.obs) to a CA-SLOPE state dict.

    The delivery-zone position is read from obs[goal_key] (default "goal", the v2 contract key).
    WARNING: obs["goal"] ANNEALS to zeros in curriculum stage 4, so batch-level shaping built from
    obs is only valid pre-anneal. For shaping that survives the anneal, use CASlopeEnvWrapper, which
    reads the unannealed env.goal_pos. See docs/ca_slope.md.
    """
    return {
        "ee_pos": obs["ee_pos"],
        "box_pos": obs["box_pos"],
        "goal_pos": obs[goal_key],
        "holding": _squeeze_flag(obs["holding"]),
        "goal_id": obs["goal_id"],
    }


class CASlopeShaper:
    """Compute the CA-SLOPE potential Phi(s) and the shaping reward F = gamma*Phi(s') - Phi(s).

    State is described by five arrays, each shaped (..., D) or (...,):
        ee_pos   (..., 3)  end-effector xyz (base frame)
        box_pos  (..., 3)  target box xyz (env-local)
        goal_pos (..., 3)  delivery zone xyz (env-local)
        holding  (...,)    bool/float, 1.0 if the target box is grasped (switches phase A<->B)
        goal_id  (..., 3)  one-hot category [fragile, regular, heavy]

    Usage::

        shaper = CASlopeShaper(category_aware=True)
        f = shaper.shaping(prev_state, next_state, done=done)   # add f to the base reward
    """

    def __init__(
        self,
        gamma: float = DEFAULT_GAMMA,
        category_gains: tuple[float, float, float] = DEFAULT_CATEGORY_GAINS,
        generic_gain: float = DEFAULT_GENERIC_GAIN,
        phase_b_offset: float = DEFAULT_PHASE_B_OFFSET,
        category_aware: bool = True,
    ):
        """Configure gains. category_aware=False = generic SLOPE (single gain) for the RQ2 control."""
        self.gamma = float(gamma)
        self.category_gains = tuple(float(g) for g in category_gains)
        self.generic_gain = float(generic_gain)
        self.phase_b_offset = float(phase_b_offset)
        self.category_aware = bool(category_aware)

    def gain(self, goal_id):
        """Per-env shaping gain (...,) selected from the one-hot goal_id.

        Selection is a dot-product with the one-hot, so it stays in the caller's array backend with
        no construction call (`goal_id[..., k]` columns * python floats). Generic mode uses one gain.
        """
        if self.category_aware:
            g0, g1, g2 = self.category_gains
        else:
            g0 = g1 = g2 = self.generic_gain
        return goal_id[..., 0] * g0 + goal_id[..., 1] * g1 + goal_id[..., 2] * g2

    def potential(self, ee_pos, box_pos, goal_pos, holding, goal_id):
        """CA-SLOPE potential Phi(s) (...,). Higher (closer to 0) = closer to task completion.

        Phase A (not holding): remaining = dist(ee, box) + phase_b_offset  (still must grasp+carry).
        Phase B (holding):     remaining = xy-dist(box, zone)              (just carry+place).
        Phi = -gain * remaining. `holding` gates the two phases via arithmetic (no torch.where), so
        the expression is identical for numpy and torch.
        """
        h = holding * 1.0  # bool -> float in both backends
        dist_ee_box = _norm_last(ee_pos - box_pos)
        dist_box_goal = _norm_last(box_pos[..., :2] - goal_pos[..., :2])  # xy only, like carry reward
        remaining = (1.0 - h) * (dist_ee_box + self.phase_b_offset) + h * dist_box_goal
        return -self.gain(goal_id) * remaining

    def shaping(self, prev_state: dict, next_state: dict, done=None):
        """Shaping reward F = gamma*Phi(s') - Phi(s), shaped like `holding` (...,).

        prev_state / next_state are dicts with keys: ee_pos, box_pos, goal_pos, holding, goal_id.
        On terminal transitions the PBRS convention sets Phi(terminal)=0, so F = -Phi(s); pass a
        boolean `done` (...,) to apply it. Gating is arithmetic to stay backend-agnostic.
        """
        phi_prev = self.potential(**prev_state)
        phi_next = self.potential(**next_state)
        f = self.gamma * phi_next - phi_prev
        if done is not None:
            d = done * 1.0
            f = f * (1.0 - d) + (-phi_prev) * d  # gamma*Phi(terminal=0) - Phi(s) = -Phi(s)
        return f

    def shaping_from_obs(self, obs: dict, next_obs: dict, done=None, goal_key: str = "goal"):
        """Shaping for a transition described by two v2 obs dicts (e.g. buffer Batch.obs/next_obs).

        Convenience over shaping(): maps obs keys via state_from_obs (goal_key -> zone, holding
        squeezed). Note the annealing caveat in state_from_obs — prefer CASlopeEnvWrapper in-env.
        """
        return self.shaping(
            state_from_obs(obs, goal_key), state_from_obs(next_obs, goal_key), done=done
        )
