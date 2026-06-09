# Bug: Ridgeback-Franka Root State Frozen (obs + reward read the fixed root)

**Date:** 2026-06-04
**File:** env/warehouse_env.py, env/warehouse_reward.py (also scripts/drive_robot.py, scripts/smoke_test.py, tests/test_env.py)
**Status:** [x] Fixed (2026-06-04)

---

## Symptom

One root cause, two faces:

1. **"Robot won't move" (teleop).** Driving `scripts/drive_robot.py` felt broken and there was no
   confident world-vs-body frame verdict.
2. **RL env silently broken.** `obs["position"]` and `obs["heading"]` never changed while the robot
   drove, and every reward/termination distance stayed at the spawn value. The policy would see a
   robot frozen at spawn → navigation is unlearnable, and `delivery_success`/`reached_goal` could
   only ever fire if the robot happened to spawn inside a zone. `tests/test_env.py` still reported
   10/10 because it only checked obs **shapes**, never that obs **change** under motion.

## Root Cause

The robot is the stock Isaac Lab `RIDGEBACK_FRANKA_PANDA_CFG`: a **fixed-root** articulation whose
base is moved by 3 dummy joints in the chain
`world → dummy_base_prismatic_x → dummy_base_prismatic_y → dummy_base_revolute_z → base_link`.

**Face A — frozen root state.** On a fixed-root articulation, `Articulation.data.root_pos_w` /
`root_quat_w` return the FIXED root link (welded to `world`). It stays at the spawn pose for the
whole episode while the chassis (`base_link`) translates/rotates via the dummy joints. The env read
the root in three places:
- `env/warehouse_env.py::robot_position`  → `root_pos_w`
- `env/warehouse_env.py::robot_heading`   → `root_quat_w`
- `env/warehouse_reward.py::_robot_xy`     → `root_pos_w` (feeds `delivery_success`,
  `distance_to_goal`, `reached_goal`, `out_of_bounds`)

So position / heading / all distances were pinned to spawn.
(`scripts/smoke_test.py` had already been fixed to read `body_pos_w[base_link]`; the env + reward
were never updated to match — that mismatch is what hid the bug.)

**Face B — world-frame translation.** The dummy prismatic joints translate along **world** axes
(they precede `revolute_z` in the chain, so they have no awareness of chassis yaw). `_base_cmd`
mapped the `(2,)` action to `[lin, 0, ang]`, so commanded "forward" always slid along world **+x**
regardless of heading. After any turn the robot could not drive toward its facing direction
(navigation stuck on a world-x rail + spin), and teleop felt wrong.

## Fix

```python
# env/warehouse_env.py + env/warehouse_reward.py — read the MOVING chassis, not the fixed root:
idx = robot.body_names.index("base_link")
robot.data.body_pos_w[:, idx]      # position  (was root_pos_w)
robot.data.body_quat_w[:, idx]     # heading   (was root_quat_w)

# env/warehouse_env.py::_base_cmd — body-frame drive on a world-frame base (project by yaw):
yaw = self._base_yaw()             # from base_link quat, cached idx
vx  = lin * torch.cos(yaw)         # → world-x velocity (prismatic_x)
vy  = lin * torch.sin(yaw)         # → world-y velocity (prismatic_y)
return torch.stack([vx, vy, ang], dim=-1)
```

- `scripts/drive_robot.py` teleop: same yaw projection (rotate `[vx, vy]` by chassis yaw).
- `scripts/smoke_test.py`: now prints `root_pos_w` vs `body_pos_w[base_link]` displacement to make
  the gotcha self-evident, plus the existing world-vs-body frame verdict.
- `tests/test_env.py`: new regression — position/heading must change under motion, and forward must
  follow heading. Drives are kept short + slow so the base stays in bounds (at full 1.5 m/s it can
  leave the room within ~1 s and auto-reset mid-measurement).

## References (per face)

- **Frozen root state:** IsaacLab Issue #1268 — "RL environment with Ridgeback Franka is not
  updating the root state" — https://github.com/isaac-sim/IsaacLab/issues/1268 ·
  corroborating Issue #2254 — https://github.com/isaac-sim/IsaacLab/issues/2254 ·
  `Articulation.data` (`root_pos_w` vs `body_pos_w`) —
  https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/assets/articulation/articulation_data.html ·
  fixed-root / `fix_root_link` —
  https://isaac-sim.github.io/IsaacLab/main/source/how-to/make_fixed_prim.html
- **World-frame drive:** IsaacLab Discussion #2664 — "Mobile Base yaw not updating xy translation"
  — https://github.com/isaac-sim/IsaacLab/discussions/2664 (maintainer: "dummy joints don't
  automatically account for child rotations — you must manually transform control vectors using
  orientation before applying them to X/Y joints") · control pattern —
  https://forums.developer.nvidia.com/t/how-to-control-clearpath-ridgebackfranka/299016

## Verification

`conda activate isaaclab; python tests/test_env.py --num_envs 1` (camera ON, driver 580.88) →
**ALL PASS (16/16)**, including the new checks:
- `Obs: position tracks forward motion (Δ>0.1m)` — Δ=0.188 m
- `Drive: forward follows heading (cos>0.6)` — cos=0.81
- `Obs: heading tracks yaw rotation (Δ>0.05)` — Δ=1.176 rad

(Before the fix, position/heading would not change → these checks fail; the suite previously only
asserted shapes, which is why 10/10 still passed with the bug present.)

## Notes / follow-up (separate from this fix)

At full 1.5 m/s the base can leave the 20×30 m room within ~1 s, and a hard velocity step appears
to produce large transients (a >6 m single-window displacement was observed = the env auto-resetting
mid-drive). This is a control/physics-tuning concern (action smoothing, effort/velocity limits),
**not** part of the root-state fix — track separately before serious training.
