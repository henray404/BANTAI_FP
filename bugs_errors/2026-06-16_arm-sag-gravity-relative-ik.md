# Arm collapses under gravity — relative-IK re-anchors to sagged pose

**Date:** 2026-06-16
**Component:** `env/warehouse_scene.py` (RIDGEBACK_FRANKA_CFG), arm control
**Severity:** High — breaks manipulation training (arm cannot hold a grasped box)

## Symptom
With the env running and **zero arm command** (`ee_dx=ee_dy=ee_dz=0`), the end-effector
sinks from the ready pose to a low gravity-rest pose on its own.

Measured via per-step teleop log (`scripts/drive_env.py --log`):
spawn `ee_z = 0.879 m` → decays smoothly → settles at `ee_z ≈ 0.043 m` in ~10 s,
all action columns `0.0`. base roll/pitch stay `0.0` (chassis level — not a base tilt).
Raising the arm (Q) works, but releasing the key makes it sag back down.

## Root cause
The arm IK action term uses `DifferentialInverseKinematicsActionCfg` with
`use_relative_mode=True` (target = current_ee + delta). Arm **gravity is enabled**, so
each control step there is a small steady-state tracking error (gravity torque / stiffness).
Relative mode re-reads the *already-sagged* current pose as its anchor each step, so the
error compounds and the arm walks down to its gravity equilibrium.

Isaac Lab's `FRANKA_PANDA_HIGH_PD_CFG` (the config used by Isaac-Lift-Cube-Franka-v0 and
Isaac-Reach-Franka-v0 — the envs CLAUDE.md says to copy for arm control) sets
`spawn.rigid_props.disable_gravity = True` precisely to avoid this: a weightless arm holds
its IK pose with no sag. Our `RIDGEBACK_FRANKA_CFG` spawn never set `disable_gravity`, so it
defaulted to gravity ON.

Reference: `C:\IsaacLab\source\isaaclab_assets\isaaclab_assets\robots\franka.py`
```python
FRANKA_PANDA_HIGH_PD_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0  # damping 80
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0   # damping 80
```

## Resolution
Add `rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)` to the
`RIDGEBACK_FRANKA_CFG.spawn` UsdFileCfg. The graspable boxes keep `disable_gravity=False`,
so a held box still has weight (matches Lift-Cube: arm gravity off, object gravity on).

Confirmed-not-bugs from the same teleop log:
- Arm X/Y/Z axes (W/S, A/D, Q/E) all respond, both directions.
- Base forward/back/yaw work; chassis stays level (the "tilt" was a chase-cam visual artifact).
