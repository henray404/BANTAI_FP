# experiments/scripted_policy.py
# Person 5 — greedy scripted policy, a PLACEHOLDER for the DreamerV3 policy (still being fixed).
#
# This is intentionally "kasaran": a hand-coded controller that drives the toy env to completion so
# the eval harness produces meaningful per-step CSV traces today. The harness takes ANY
# callable policy(obs) -> action(6,), so when P3's DreamerV3 actor is ready, pass it instead and the
# CSV/metrics pipeline is unchanged. Not a learned policy — do not report it as a baseline number.

"""Greedy approach->grasp->carry->place controller used until the learned policy lands."""

from __future__ import annotations

import numpy as np


class ScriptedPickupPolicy:
    """Heuristic controller over the toy obs dict. Stateless except for a tiny grasp dwell counter."""

    def __init__(self, base_gain: float = 1.0, ee_gain: float = 1.0):
        """Gains scale the proportional commands; defaults work with toy_pickup_env kinematics."""
        self.base_gain = base_gain
        self.ee_gain = ee_gain

    def reset(self) -> None:
        """Clear per-episode state."""
        self._dwell = 0

    def __call__(self, obs: dict) -> np.ndarray:
        """Map an obs dict to a (6,) action in [-1, 1]."""
        base_xy = np.asarray(obs["position"], dtype=np.float64)
        yaw = float(obs["heading_yaw"])
        ee = np.asarray(obs["ee_pos"], dtype=np.float64)
        box = np.asarray(obs["box_pos"], dtype=np.float64)
        goal = np.asarray(obs["goal_pos"], dtype=np.float64)
        holding = bool(obs["holding"])

        target_xy = goal[:2] if holding else box[:2]

        # ── Base: steer toward, and drive at, the active target ──────────
        to_target = target_xy - base_xy
        dist_xy = float(np.linalg.norm(to_target))
        desired_yaw = np.arctan2(to_target[1], to_target[0])
        yaw_err = (desired_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        base_ang = float(np.clip(yaw_err * 1.5, -1.0, 1.0))
        # Only drive forward once roughly facing the target; ease off when close.
        aligned = abs(yaw_err) < 0.5
        base_lin = float(np.clip(dist_xy, 0.0, 1.0)) if aligned and dist_xy > 0.4 else 0.0

        # ── End-effector: reach toward box (Phase A) / hold high over goal (Phase B) ──
        if holding:
            ee_target = np.array([goal[0], goal[1], 0.6])
        else:
            ee_target = box.copy()
        ee_err = ee_target - ee
        # ee delta command is in base frame; rotate the world error back by -yaw.
        c, s = np.cos(-yaw), np.sin(-yaw)
        ee_dx = c * ee_err[0] - s * ee_err[1]
        ee_dy = s * ee_err[0] + c * ee_err[1]
        ee_cmd = np.clip(np.array([ee_dx, ee_dy, ee_err[2]]) * self.ee_gain * 5.0, -1.0, 1.0)

        # ── Gripper: close when lined up on the box; open when delivered over the zone ──
        grip = 1.0  # open by default
        if not holding:
            near_box = np.linalg.norm(ee[:2] - box[:2]) < 0.18 and abs(ee[2] - box[2]) < 0.20
            if near_box:
                grip = -1.0  # close to grasp
        else:
            over_zone = np.linalg.norm(box[:2] - goal[:2]) < 1.4
            grip = 1.0 if over_zone else -1.0  # keep closed until over the zone, then release

        return np.array([base_lin, base_ang, ee_cmd[0], ee_cmd[1], ee_cmd[2], grip], dtype=np.float32)
