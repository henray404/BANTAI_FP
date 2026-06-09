# ADR 0002 — Ridgeback-Franka Robot with Active Arm

- **Status:** Accepted
- **Date:** 2026-06-08 (robot swap 2026-06-01; arm activated in pickup redesign 2026-06-08)
- **Spec:** `docs/superpowers/specs/2026-06-08-pure-dl-pickup-redesign.md` §3, §5

## Context

Earlier scene used a Carter/Jetbot wheeled base — navigation only, no manipulator. The redesigned task requires picking a box off a shelf and placing it in a zone, which a navigation-only base cannot do. A mobile manipulator is required: holonomic base for warehouse navigation plus an arm with reach into the rack.

Hand-rolling inverse kinematics for a 7-DOF arm is error-prone and out of scope for a DL course project.

## Decision

Use the **Ridgeback-Franka** mobile manipulator (Isaac 5.1 Nucleus USD): Clearpath holonomic base + Franka Panda 7-DOF arm + parallel gripper. The Franka arm is **active** (was tucked).

- **Base** velocity control on 3 dummy joints (prismatic x/y, revolute z), holonomic; action drives x(lin) + z(ang), y=0.
- **Arm** position control via `isaaclab.controllers.DifferentialIKController`. Policy commands EE position delta `(ee_dx, ee_dy, ee_dz)` in base frame; orientation held fixed top-down. Do **not** hand-roll IK.
- Reference working envs to copy: `Isaac-Reach-Franka-v0`, `Isaac-Lift-Cube-Franka-v0`.
- cuRobo (GPU motion planner) available later if collision-aware reaches needed — out of scope v1.

## Consequences

- Action space grows from base-only to **(6,)**: `[base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper]`.
- Obs gains manipulation keys: `ee_pos`, `gripper`, `holding`, `box_pos`.
- Boxes must spawn within Franka reach (~0.85m) from a feasible base pose — a hard scene constraint (see ADR-0003 box reduction).
- P1 owns arm IK wiring; pairs with P4 on grasp.
- VRAM must be re-measured with active arm IK + ~18 boxes on RTX 5050 8GB.
