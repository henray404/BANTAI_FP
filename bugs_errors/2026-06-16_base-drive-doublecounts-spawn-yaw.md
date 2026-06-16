# Base drive direction off by the spawn yaw — `_base_cmd` double-counts root rotation

**Date:** 2026-06-16
**Component:** `env/warehouse_env.py` (`WarehouseGymEnv._base_cmd` / `_base_yaw`)
**Severity:** High — robot moves at a per-episode angular offset from where it faces; breaks
navigation and corrupts the relation between `heading`/onboard camera and actual motion.

## Symptom
Pressing "forward" drives the robot at a constant angular offset from its visual heading. The
offset looked like ~30° by eye but is actually **episode-dependent** (equals the random spawn yaw).

## Measurement (teleop CSV, `--log`)
Pure-forward segments, no yaw mixed in:

| Segment | `base_yaw` (reported) | motion dir `atan2(Δwy, Δwx)` | motion − yaw |
|---|---|---|---|
| 1 (revolute_z = 0) | 142.37° | −75.3° | +142.4° |
| 2 (after rotating) | −123.08° | +19.3° | +142.4° |

`motion_dir = base_yaw + 142.37°`, and **142.37° = the spawn yaw**. Constant across both
segments → the spawn root yaw is being added twice.

## Root cause
Kinematic chain: `world → prismatic_x → prismatic_y → revolute_z → base_link`. The dummy
prismatic joints translate in the **articulation-root frame**, and the root pose is yaw-randomized
on reset (`pose_range yaw (-π, π)`). So the prismatic axes are already rotated by `root_yaw`.

`_base_cmd` projected the linear command by the **absolute base_link world yaw**
(`= root_yaw + revolute_z`) and fed it to the prismatic joints, which re-apply `root_yaw`:
`motion_world = root_yaw + (root_yaw + revolute_z)` → off by `root_yaw`. The earlier #2664 fix
correctly identified world-vs-body but used the wrong yaw reference.

## Resolution
Project the linear command by the **`dummy_base_revolute_z_joint` angle** (chassis yaw *relative
to* the root/prismatic frame), not the absolute world yaw:
`motion_world = root_yaw + revolute_z = base_link world yaw` = the facing direction. Correct, and
now consistent with the `heading` obs (absolute world yaw) and the base-mounted onboard camera.

Only manifested clearly because spawn yaw is randomized; an episode that happened to spawn near
yaw 0 would have looked fine.
