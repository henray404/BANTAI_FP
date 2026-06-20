# experiments/toy_pickup_env.py
# Person 5 — pure-numpy KINEMATIC stand-in for WarehouseGymEnv, for headless eval on a Mac.
#
# WHY: Isaac Lab does not run on macOS, and the DreamerV3 policy is still being fixed. To produce
# CA-SLOPE performance traces NOW, this env reproduces the obs/action CONTRACT and the staged-reward
# SHAPE (env/reward_pickup.py) without physics — pure point-mass kinematics. It is deliberately a
# "kasaran" (rough) approximation: good enough to exercise CA-SLOPE shaping and emit step-by-step
# CSV traces, NOT a physics replacement. Swap it for the real WarehouseGymEnv (same step/obs API)
# once Isaac is available — the harness only needs reset()/step() and the obs keys below.
#
# Contract reproduced (subset of the 9-key obs that CA-SLOPE + the harness need):
#   obs = position(2), heading_yaw, ee_pos(3), box_pos(3), goal_pos(3), goal_id(3), holding, gripper
# Action: [base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper], shape (6,), all in [-1, 1].

"""Headless numpy kinematic stand-in reproducing the warehouse pickup obs/action/reward contract."""

from __future__ import annotations

import numpy as np

from experiments.scenarios import Scenario

# Per-step kinematic gains (rough, tuned so a greedy policy finishes in a few hundred steps).
BASE_SPEED = 0.30        # metres per unit base_lin per step
TURN_SPEED = 0.20        # radians per unit base_ang per step
EE_REACH = 0.04          # metres per unit ee_d* per step
ARM_RADIUS = 0.85        # Franka reach from base (CONTEXT.md); ee is clamped within this of the base
EE_REST = np.array([0.45, 0.0, 0.60])  # ee home offset from base (base frame), tucked-ish

GRASP_XY_THRESH = 0.18   # ee within this xy of the box (and low enough) grasps when gripper closes
GRASP_Z_THRESH = 0.20
DELIVER_RADIUS = 1.5     # env/reward_pickup.DELIVER_RADIUS_M — box xy within this of zone = delivered

# Staged-reward weights — mirror env/warehouse_reward + reward_pickup magnitudes (CONTEXT.md table).
R_GRASP = 5.0
R_DELIVER = 10.0
R_DIST = 0.01            # * active-phase distance, applied negative (dense shaping)
R_TIME = 0.005          # per-step time penalty
R_DROP = 2.0            # box released outside any zone while carrying


class ToyPickupEnv:
    """Single-env, pure-numpy kinematic pickup. Mirrors the WarehouseGymEnv step/obs/reward shape."""

    def __init__(self, max_steps: int = 600):
        """max_steps caps an episode (env time_limit ~= 600 in training configs)."""
        self.max_steps = int(max_steps)
        self._scn: Scenario | None = None
        self.rng = np.random.default_rng(0)

    # ── lifecycle ────────────────────────────────────────────────────────
    def reset(self, scenario: Scenario, seed: int = 0):
        """Place the robot at the scenario spawn; return (obs, info)."""
        self._scn = scenario
        self.rng = np.random.default_rng(seed)
        self.t = 0
        # Small per-seed spawn jitter so seeds differ (mirrors random spawn yaw/pos in receiving).
        jitter = self.rng.uniform(-0.5, 0.5, size=2)
        self.base_xy = np.array(scenario.spawn_xy, dtype=np.float64) + jitter
        self.base_yaw = float(self.rng.uniform(-np.pi, np.pi))
        self.box_xyz = np.array(scenario.box_xyz, dtype=np.float64)
        self.goal_xyz = np.array(scenario.goal_xyz, dtype=np.float64)
        self.goal_id = np.array(scenario.goal_id, dtype=np.float64)
        self.holding = False
        self.gripper = 1.0  # 1=open, like the contract (gripper>0 open)
        self.ee_pos = self._ee_home()
        return self._obs(), {"phase": "A"}

    def _ee_home(self) -> np.ndarray:
        """End-effector world xyz when parked at rest relative to the (yawed) base."""
        c, s = np.cos(self.base_yaw), np.sin(self.base_yaw)
        off = EE_REST
        world_xy = self.base_xy + np.array([c * off[0] - s * off[1], s * off[0] + c * off[1]])
        return np.array([world_xy[0], world_xy[1], off[2]])

    # ── step ─────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        """Apply a (6,) action; return (obs, base_reward, terminated, truncated, info)."""
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(6), -1.0, 1.0)
        base_lin, base_ang, ee_dx, ee_dy, ee_dz, grip_cmd = a
        self.t += 1

        # Base: holonomic-ish — drive forward along heading, yaw in place.
        self.base_yaw = (self.base_yaw + base_ang * TURN_SPEED + np.pi) % (2 * np.pi) - np.pi
        c, s = np.cos(self.base_yaw), np.sin(self.base_yaw)
        self.base_xy = self.base_xy + base_lin * BASE_SPEED * np.array([c, s])

        # End-effector: integrate base-frame delta into world, then clamp within Franka reach.
        d_world = np.array([c * ee_dx - s * ee_dy, s * ee_dx + c * ee_dy, ee_dz]) * EE_REACH
        self.ee_pos = self.ee_pos + d_world
        # Base drift moves the ee home with it; re-anchor xy reach to the base each step.
        base_xyz = np.array([self.base_xy[0], self.base_xy[1], 0.0])
        rel = self.ee_pos - base_xyz
        dist = np.linalg.norm(rel)
        if dist > ARM_RADIUS:
            rel = rel / dist * ARM_RADIUS
            self.ee_pos = base_xyz + rel
        self.ee_pos[2] = float(np.clip(self.ee_pos[2], 0.05, 1.35))  # floor / reach ceiling

        prev_holding = self.holding
        grasp_event = drop_event = deliver_event = False

        closing = grip_cmd <= 0.0
        self.gripper = 0.0 if closing else 1.0

        if self.holding:
            # Carry: box rides with the end-effector.
            self.box_xyz = self.ee_pos.copy()
            in_zone = np.linalg.norm(self.box_xyz[:2] - self.goal_xyz[:2]) < DELIVER_RADIUS
            if not closing:  # gripper opened -> release
                self.holding = False
                if in_zone:
                    deliver_event = True
                else:
                    drop_event = True
        else:
            # Phase A: grasp if the gripper closes on the box (close enough in xy and z).
            dxy = np.linalg.norm(self.ee_pos[:2] - self.box_xyz[:2])
            dz = abs(self.ee_pos[2] - self.box_xyz[2])
            if closing and dxy < GRASP_XY_THRESH and dz < GRASP_Z_THRESH:
                self.holding = True
                grasp_event = True

        reward, terminated = self._reward(prev_holding, grasp_event, deliver_event, drop_event)
        truncated = self.t >= self.max_steps
        info = {
            "phase": "B" if self.holding else "A",
            "grasp_event": grasp_event,
            "deliver_event": deliver_event,
            "drop_event": drop_event,
            "success": deliver_event,
        }
        return self._obs(), reward, terminated, truncated, info

    # ── reward (mirrors staged reward magnitudes) ────────────────────────
    def _reward(self, prev_holding, grasp_event, deliver_event, drop_event):
        """Staged reward + terminal flag, matching the CONTEXT.md reward table shape."""
        r = -R_TIME  # time_penalty every step
        if self.holding or prev_holding:
            active_dist = np.linalg.norm(self.box_xyz[:2] - self.goal_xyz[:2])  # Phase B: box->zone
        else:
            active_dist = np.linalg.norm(self.ee_pos - self.box_xyz)            # Phase A: ee->box
        r -= R_DIST * active_dist
        if grasp_event:
            r += R_GRASP
        if deliver_event:
            r += R_DELIVER
        if drop_event:
            r -= R_DROP
        return float(r), bool(deliver_event)

    # ── obs ──────────────────────────────────────────────────────────────
    def _obs(self) -> dict:
        """Subset of the 9-key contract that CA-SLOPE + the harness consume (env-local frame)."""
        return {
            "position": self.base_xy.copy(),
            "heading_yaw": float(self.base_yaw),
            "ee_pos": self.ee_pos.copy(),
            "box_pos": self.box_xyz.copy(),
            "goal_pos": self.goal_xyz.copy(),
            "goal_id": self.goal_id.copy(),
            "holding": bool(self.holding),
            "gripper": float(self.gripper),
        }
