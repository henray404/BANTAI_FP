# Robot Discovery: Wheeled AMR for Warehouse Env

**Date:** 2026-05-22
**File:** env/warehouse_scene.py, env/warehouse_env.py
**Status:** [x] Decided

---

## Goal
Pick a wheeled differential-drive robot for the warehouse env. Interface contract requires
`action_space = Box(-1, 1, shape=(2,))` representing `[linear_vel, angular_vel]`.

## What I Checked
Scanned `C:\IsaacLab\source\isaaclab_assets\isaaclab_assets\robots\*.py`:
- agibot, agility, allegro, ant, anymal, arl_robot_1, cartpole, cart_double_pendulum,
  cassie, fourier, franka, galbot, humanoid, humanoid_28, kinova, kuka_allegro,
  openarm, pick_and_place, quadcopter, ridgeback_franka, sawyer, shadow_hand,
  spot, unitree, universal_robots

Searched for "carter | jetbot | nova | transporter | differential | wheel" via Grep —
no standalone wheeled robot config (only `ridgeback_franka` which bundles a Franka arm
and requires `ISAAC_NUCLEUS_DIR` USD).

Tutorial `C:\IsaacLab\scripts\tutorials\01_assets\add_new_robot.py` defines a working
`JETBOT_CONFIG` using the Nucleus USD:

```python
JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)
```

Joints driven by `[left_wheel, right_wheel]` velocities — confirmed by tutorial driving
`scene["Jetbot"].set_joint_velocity_target(torch.Tensor([[10.0, 10.0]]))`.

## Decision
Use **Jetbot via Nucleus USD** as Person 1 placeholder AMR.

Reasoning:
- Only wheeled robot with a tested config in this Isaac Lab install.
- Small footprint fits a 7x7m warehouse cell.
- 2 wheel joints map cleanly to differential drive.
- "No external USD" rule in the prompt applies to **scene props** (racks/items/zones).
  Robot section explicitly says "Use ROBOT_CFG.replace(prim_path=...)" — Nucleus USD is
  the only path that yields a working articulated wheeled robot here.

## Joint Names
Resolved at first sim play via `joint_names_expr=[".*"]`. Tutorial confirms 2 joints,
order `[left, right]`. Differential-drive mapping in Gym wrapper:

```
v_left  = (linear_vel - 0.5 * wheel_base * angular_vel) / wheel_radius
v_right = (linear_vel + 0.5 * wheel_base * angular_vel) / wheel_radius
```

## Camera
Onboard `CameraCfg` spawned under `{ENV_REGEX_NS}/Robot/chassis/camera`. 64x64 RGB,
ROS convention, ~0.3m forward offset.

## Risk
- Requires Omniverse Nucleus online (default localhost on Windows install).
- If Nucleus unavailable, fallback = build wheeled bot from primitive cylinders + joints
  (not implemented — out of Person 1 scope this sprint).
