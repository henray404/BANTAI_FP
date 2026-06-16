# Robot Specification — Ridgeback-Franka Mobile Manipulator

> Canonical numeric source: `configs/env_config.yaml` (`robot:` block). Python mirrors it in
> `env/warehouse_scene.py` (`RIDGEBACK_FRANKA_CFG`) and `env/warehouse_env.py` (`ActionsCfg`).
> Update all three when a value changes.

## Overview
| Field | Value |
|---|---|
| Platform | Ridgeback-Franka (Clearpath holonomic base + Franka Panda 7-DOF arm + 2-finger gripper) |
| USD | `ISAAC_NUCLEUS_DIR/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd` (Isaac 5.1 Nucleus) |
| Replaced | Carter/Jetbot (2026-06-01) — needed an arm for the pickup task |
| Total joints | 12 = 3 base + 7 arm + 2 finger |

## Joints & control
```
Base (velocity ctrl, holonomic): dummy_base_prismatic_x_joint, dummy_base_prismatic_y_joint,
                                 dummy_base_revolute_z_joint
Arm  (position ctrl via DifferentialIK): panda_joint1..7
Grip (position ctrl): panda_finger_joint1/2
```
- **Base** — 3 dummy joints (no wheel kinematics). Action drives x(linear) + z(angular); y forced 0
  (no strafe). `_base_cmd` projects the linear command by chassis yaw so "forward" follows heading
  (world-frame prismatic joints would otherwise always slide along world +x — IsaacLab #2664).
- **Arm** — `DifferentialIKControllerCfg(command_type="position", use_relative_mode=True, ik_method="dls")`
  on `panda_joint.*`, EE body `panda_hand`. EE orientation held fixed top-down (gripper points down).
- **Gripper** — binary: open `0.035`, closed `0.0` on the two finger joints.

## Action space
```python
action = [base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper]   # shape (6,), values in [-1, 1]
```
| Axis | Meaning | Scale |
|---|---|---|
| `base_lin` | base forward speed (body frame) | × `max_lin_speed = 1.5` m/s |
| `base_ang` | base yaw rate | × `max_ang_speed = 1.5` rad/s |
| `ee_dx,dy,dz` | EE Cartesian position **delta**, base frame | `0.05` m travel per step at action=1.0 |
| `gripper` | >0 open, ≤0 close | binary |

External (6,) → internal (7,) joint action: `base_vel(3) + arm_ik(3) + gripper(1)`
(`WarehouseGymEnv.step` → `split_action` → `_base_cmd`).

## Initial (ready) arm pose
```
panda_joint1=0.0  joint2=-0.569  joint3=0.0  joint4=-2.810
joint5=0.0  joint6=2.0  joint7=0.741   fingers=0.035 (open)
```

## Actuators (ImplicitActuatorCfg)
| Group | Joints | Stiffness | Damping | Effort limit |
|---|---|---|---|---|
| base | `dummy_base_.*` | 0.0 | 1.0e5 | 1000 |
| panda_shoulder | `panda_joint1-4` | 800.0 | 40.0 | 87.0 |
| panda_forearm | `panda_joint5-7` | 800.0 | 40.0 | 12.0 |
| panda_hand | `panda_finger_joint.*` | 1.0e5 | 1.0e3 | 200.0 |

Articulation props: `enabled_self_collisions=False`, `solver_position_iteration_count=12`,
`solver_velocity_iteration_count=1`, `activate_contact_sensors=True`.

## Gravity (important — fixed 2026-06-16)
The arm spawns with **`rigid_props.disable_gravity = True`**. Without it, relative-mode IK
re-anchors to the gravity-sagged pose each step and the arm collapses from `ee_z≈0.88` to
`≈0.04` on its own. Disabling gravity (matching Isaac's `FRANKA_PANDA_HIGH_PD_CFG`, used by
Lift-Cube/Reach) lets the arm hold its IK target. Graspable boxes keep gravity, so a held box
still has weight. See `bugs_errors/2026-06-16_arm-sag-gravity-relative-ik.md`.

## Onboard camera (proprioceptive sensor)
| Field | Value |
|---|---|
| Type | **TiledCameraCfg** (CameraCfg crashes on RTX 5050 Blackwell) |
| Mount | `offset_pos=[0.35, 0.0, 0.55]` front-top of base, forward-facing |
| Resolution | 64×64 RGB |
| Focal length | 18 mm → HFOV ≈ 60° (RealSense D435-like) |
| Update | every 0.1 s (10 Hz) |

## Gotchas
- **Fixed-root articulation**: `root_pos_w` stays at spawn while the chassis moves. Read
  `body_pos_w["base_link"]` for the live base pose, NOT `root_pos_w` (IsaacLab #1268).
- Pin NVIDIA driver **580.88** (591.x/595.x reintroduce the Blackwell camera SDP crash).
- Each finished Isaac run leaves a zombie `python.exe` (Blackwell `close()` hang) — kill between runs.

## Teleop (manual drive) — `scripts/drive_env.py`
| Action | Key |
|---|---|
| Base fwd/back | ↑ / ↓ |
| Base yaw | Z / X |
| Arm EE ±x | W / S |
| Arm EE ±y | A / D |
| Arm EE ±z | Q / E |
| Gripper toggle | K |
| Reset arm cmd | L |
Flags: `--ee_sens`, `--chase` (3rd-person follow cam), `--log <csv>` (per-step diagnostic).
