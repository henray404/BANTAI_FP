# buffer/visual_her.py
# P3 (Jeremy) — Visual HER relabeling for the warehouse pickup task.
#
# Standard HER relabels by the position the robot reached.
# Visual HER additionally relabels `goal_id` to match the category
# the robot actually interacted with (box grasped / zone reached),
# giving positive signal to failed trajectories where the robot
# approached a box or zone of a different category.
#
# Zone ↔ category mapping (from env_config.yaml, DO NOT change without P1):
#   index 0 = orange = fragile → zone_A at (-6, -12, 0.01)
#   index 1 = cyan   = regular → zone_B at ( 0, -12, 0.01)
#   index 2 = purple = heavy   → zone_C at ( 6, -12, 0.01)
#
# Spec (pembagian_tugas.md §P3 Visual HER):
#   Relabel episode gagal (box yang ke-grasp / zona yang dicapai)
#   seakan itu goal_id yang diperintah → rollout gagal tetap kasih sinyal positif.

"""Visual HER relabeling function factory for goal_id-based warehouse task."""

from __future__ import annotations

from typing import Callable

import numpy as np

# Zone xyz positions matching env_config.yaml (env-local coords).
# Index matches goal_id one-hot: 0=orange, 1=cyan, 2=purple.
ZONE_POSITIONS: np.ndarray = np.array(
    [[-6.0, -12.0, 0.01],   # zone_A (orange / fragile)
     [ 0.0, -12.0, 0.01],   # zone_B (cyan   / regular)
     [ 6.0, -12.0, 0.01]],  # zone_C (purple / heavy)
    dtype=np.float32,
)

# One-hot vectors for goal_id, index matches ZONE_POSITIONS.
ZONE_GOAL_IDS: np.ndarray = np.eye(3, dtype=np.float32)  # shape (3, 3)

# Delivery success threshold (matches warehouse_reward.py delivery_success threshold).
_DELIVERY_THRESHOLD_M: float = 1.5


def make_visual_her_fn(
    zone_positions: np.ndarray = ZONE_POSITIONS,
    success_reward: float = 10.0,
    her_ratio: float = 0.5,
) -> Callable[[list[dict]], list[dict]]:
    """Build a Visual HER relabeling function for EpisodeBuffer.

    Strategy:
        1. If the robot grasped the box at any point (holding>=0.5 in next_obs):
           - Find the delivery zone the robot got closest to after grasping.
           - Relabel all post-grasp steps: set `goal` to that zone's xyz and
             `goal_id` to that zone's one-hot.
           - Give `success_reward` on the final step.
        2. If the robot never held the box, return [] — no relabeling.

    The "visual" aspect: we inspect what the robot physically achieved
    (held box + approached zone) rather than just its final XY position.

    Args:
        zone_positions: (3, 3) float32 array of zone xyz for goal_id indices 0/1/2.
        success_reward: Reward injected on the relabeled final transition.
        her_ratio:      Fraction of trajectories that get relabeled. Uses numpy RNG.

    Returns:
        fn(trajectory: list[dict]) -> list[dict]
        Each output dict: {"obs": dict, "action": ndarray,
                           "reward": float, "next_obs": dict, "done": bool}
    """
    zone_pos = np.asarray(zone_positions, dtype=np.float32)

    def relabel_fn(trajectory: list[dict]) -> list[dict]:
        if not trajectory:
            return []

        # Stochastic gating.
        if np.random.rand() > her_ratio:
            return []

        # Find first step where box was grasped (holding transitions → 1).
        grasp_idx: int | None = None
        for i, step in enumerate(trajectory):
            holding_val = np.asarray(
                step["next_obs"].get("holding", [0.0])
            ).reshape(-1)[0]
            if holding_val >= 0.5:
                grasp_idx = i
                break

        if grasp_idx is None:
            return []  # Robot never grasped; no positive signal to extract.

        # Find which zone robot got closest to after grasping.
        min_dist = np.inf
        nearest_zone_idx = 0
        for step in trajectory[grasp_idx:]:
            robot_pos = np.asarray(
                step["next_obs"].get("position", [0.0, 0.0, 0.0])
            ).reshape(-1)
            for zi, zp in enumerate(zone_pos):
                d = float(np.linalg.norm(robot_pos[:2] - zp[:2]))
                if d < min_dist:
                    min_dist = d
                    nearest_zone_idx = zi

        new_goal = zone_pos[nearest_zone_idx].copy()          # (3,)
        new_goal_id = ZONE_GOAL_IDS[nearest_zone_idx].copy()  # (3,) one-hot

        # Relabel all post-grasp steps with the achieved goal.
        relabeled: list[dict] = []
        post_grasp = trajectory[grasp_idx:]
        n = len(post_grasp)

        for j, step in enumerate(post_grasp):
            is_last = j == n - 1

            new_obs = {**step["obs"],
                       "goal": new_goal,
                       "goal_id": new_goal_id}
            new_next_obs = {**step["next_obs"],
                            "goal": new_goal,
                            "goal_id": new_goal_id}

            relabeled.append({
                "obs": new_obs,
                "action": step["action"],
                "reward": success_reward if is_last else step["reward"],
                "next_obs": new_next_obs,
                "done": True if is_last else step["done"],
            })

        return relabeled

    return relabel_fn
