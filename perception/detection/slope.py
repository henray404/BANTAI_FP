# perception/detection/slope.py
# Person 3 — Category-Aware SLOPE reward shaping (NOVELTY, Week 6).
#
# SLOPE (Li et al. 2026): potential-based reward shaping that replaces a binary
# reward with a gradual landscape. Standard SLOPE = one landscape for all goals.
# Category-Aware SLOPE = a SEPARATE potential per item category, conditioned on the
# visual category that is the current goal. The detector (model.BoxDetector) tells
# us where the goal-category box is in view; the potential rewards reducing the gap
# to it. Auxiliary signal added to the env reward in the training loop.
#
# STATUS: skeleton + potential-based shaping interface. Quantile reward head (QCE)
# is left as a TODO for the full SLOPE; the generic potential below is the
# "SLOPE generic first" milestone.
#
# CANONICAL CA-SLOPE (2026-06-21): the pure-DL category-aware reward used by the ablation
# (configs #4/#6) lives in `reward/ca_slope.py` (CASlopeShaper) + `reward/ca_slope_wrapper.py`
# (CASlopeEnvWrapper) — adopted from a teammate's branch; it has per-category gains, a
# backend-agnostic (numpy+torch) potential, and proper PBRS terminal handling. The
# detector-based `potential()` below is the OLD vision framing and is NOT used by the
# pure-DL experiments. Prefer `reward/ca_slope.py`.

"""Category-Aware SLOPE potential-based reward shaping (legacy vision skeleton)."""

from __future__ import annotations

CATEGORIES = ("fragile", "regular", "heavy")


def potential(detections, goal_category: str, gamma: float = 0.997) -> float:
    """Potential Φ for the current frame given detections + the goal category.

    Higher when the goal-category box is large + centered (robot is close + facing).
    Returns Φ in [0, 1]. The training loop forms the shaped reward as the
    potential-based difference F = γ·Φ(s') − Φ(s), which is policy-invariant.
    """
    best = 0.0
    for cat, conf, (cx, cy, w, h) in detections:
        if cat != goal_category:
            continue
        # area ∝ proximity; centering ∝ alignment. Combine, weight by confidence.
        area = max(0.0, min(1.0, w * h * 4.0))         # normalized; ~0.25 area → 1.0
        centered = 1.0 - min(1.0, (abs(cx - 0.5) + abs(cy - 0.5)))
        best = max(best, conf * 0.5 * (area + centered))
    return float(best)


def shaped_reward(phi_prev: float, phi_next: float, gamma: float = 0.997,
                  scale: float = 1.0) -> float:
    """Potential-based shaping term F = scale·(γ·Φ' − Φ). Add to env reward."""
    return scale * (gamma * phi_next - phi_prev)


class CategoryAwareSlope:
    """Stateful SLOPE shaper: tracks previous potential per env.

    Wire in the training loop:
        slope = CategoryAwareSlope(detector)
        ...
        aux = slope.step(image, goal_category)   # add `aux` to the env reward
        slope.reset()                            # on episode boundary
    """

    def __init__(self, detector, gamma: float = 0.997, scale: float = 1.0):
        """Hold a BoxDetector + shaping params."""
        self.detector = detector
        self.gamma = gamma
        self.scale = scale
        self._phi_prev = 0.0

    def reset(self) -> None:
        """Clear potential at an episode boundary."""
        self._phi_prev = 0.0

    def step(self, image, goal_category: str) -> float:
        """Detect, compute potential, return the shaped reward delta."""
        det = self.detector.detect(image)
        phi = potential(det, goal_category, self.gamma)
        f = shaped_reward(self._phi_prev, phi, self.gamma, self.scale)
        self._phi_prev = phi
        return f
