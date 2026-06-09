# Bug: Ridgeback-Franka loads FLOATING — base never moves (Isaac 5.1)

**Date:** 2026-06-09
**File:** env/warehouse_scene.py (RIDGEBACK_FRANKA_CFG); affects scripts/drive_robot.py,
scripts/smoke_test.py, env/warehouse_env.py, all nav.
**Status:** [x] Fixed (2026-06-09) — weld `world` link at spawn.

---

## Symptom

Robot would not move in `scripts/drive_robot.py` despite arrow keys producing correct commands.
`scripts/smoke_test.py` reported `INCONCLUSIVE — chassis barely moved` even though the base joint
reached the commanded velocity.

## Root Cause

The Isaac Sim 5.1 `ridgeback_franka.usd` ships **no base anchor**. The articulation loads as a
**floating base**, and PhysX auto-roots it at `panda_link2` (an arm link). The dummy holonomic
joints (`dummy_base_prismatic_x/y`, `dummy_base_revolute_z`) therefore do NOT translate the chassis
in world space — driving them just rearranges the floating articulation internally: the light
`world` leaf link slides while the heavy chassis (`base_link`) stays put.

This is NOT our code. Proven by spawning the **official** `RIDGEBACK_FRANKA_PANDA_CFG` flat (no
InteractiveScene, no cloning) — same result. IsaacLab #1268 / #2254. Isaac Lab ships no task env
that uses the Ridgeback, only a demo that has the same defect.

### Evidence (smoke / diag, before fix)

```
is_fixed_base = False   root body = panda_link2
base joint_vel after drive = [1.000, 0, 0]        # actuator works
base joint_pos delta       = [1.000, 0, 0]        # DOF advances a full 1.0 m
base_link world-x disp     = 0.000 m              # but chassis does NOT move
world link world-x         = -1.000 m             # the `world` leaf absorbs the motion
```

(The earlier 2026-06-04 "16/16 PASS" was a reset transient + shape-only tests; the base never
actually translated. This is a distinct, deeper bug from 2026-06-04_ridgeback-root-state-frozen.md,
which only fixed the obs reading the fixed root.)

## Fix

Weld the robot's `world` link to the stage world frame with a `UsdPhysics.FixedJoint` at spawn,
BEFORE the physics view is created. This makes the articulation fixed-base, re-roots it at `world`,
and the dummy holonomic joints then translate `base_link` in world space.

`env/warehouse_scene.py`:
- `_weld_robot_world_links()` — scans the stage, welds each `<...>/Robot/world` link (handles
  1..N envs, idempotent).
- `_spawn_ridgeback_welded()` — wraps `spawn_from_usd`, then welds.
- `RIDGEBACK_FRANKA_CFG.spawn.func = _spawn_ridgeback_welded`.

`isaaclab.sim.schemas.modify_articulation_root_properties` `fix_root_link=True` does NOT work here:
it raises `NotImplementedError` because the articulation root prim is an Xform without RigidBodyAPI
and there is no existing fixed joint to toggle. Adding our own fixed joint is the working route.

### Evidence (after fix)

```
created fixed joint /World/Robot/base_world_anchor -> /World/Robot/world
is_fixed_base = True     root body = world
joint order: base joints now [0,1,2] (matches official check script)
base_link world-x disp = +0.91 m   # chassis moves; 0.91 vs 1.0 = accel ramp-up
world link world-x     = +0.00 m   # `world` stays pinned
```

## References

- IsaacLab #1268 — Ridgeback Franka root state not updating —
  https://github.com/isaac-sim/IsaacLab/issues/1268
- IsaacLab #2254 (corroborating)
- `isaaclab/sim/schemas/schemas.py::modify_articulation_root_properties` (fix_root_link logic +
  the `find_global_fixed_joint_prim` "enable existing joint" path)
- Official demo: `IsaacLab/source/isaaclab/test/assets/check_ridgeback_franka.py`

## Follow-up

- `tests/test_env.py`: add a regression that the chassis (`body_pos_w[base_link]`) translates > 0.1 m
  under a sustained forward command, and that `is_fixed_base` is True. Shape-only tests hid this.
- Remove temp diagnostics from `scripts/smoke_test.py` and delete `scripts/_diag_ridgeback.py` once
  the env-side fix is verified.
- Re-check the obs `_base_cmd` yaw projection still holds now that the base actually moves.
