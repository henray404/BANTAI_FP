# Bug: JETBOT_CFG not in isaaclab_assets

**Date:** 2026-05-15  
**File:** env/warehouse_env.py:35  
**Status:** [x] Workaround

---

## Error Message
```
ModuleNotFoundError: No module named 'isaaclab_assets.robots.jetbot'
```

## What I Was Doing
Running `python tests/test_env.py` after fixing double-AppLauncher crash.

## Root Cause
Isaac Lab 0.54.3 / 5.x does NOT ship a jetbot config in `isaaclab_assets.robots`.
Available robots: agibot, agility, allegro, ant, anymal, cartpole, cassie,
fourier, franka, galbot, humanoid, kinova, kuka_allegro, openarm,
pick_and_place, quadcopter, ridgeback_franka, sawyer, shadow_hand, spot,
unitree, universal_robots. No standalone wheeled differential-drive bot.

## Fix
Swap placeholder to `CARTPOLE_CFG` for env pipeline sanity check.
Update `ActionsCfg.joint_names` to `["slider_to_cart"]`.
Update `ObservationsCfg` to use `mdp.joint_pos_rel` / `mdp.joint_vel_rel`
(cartpole has no floating base — `base_pos_w` would fail).

Real wheeled AMR config must be added later (custom USD or ridgeback base).

## Notes
For future wheeled robot: either
- Use custom USD via `UsdFileCfg` + manual `ArticulationCfg`
- Use `RIDGEBACK_FRANKA_PANDA_CFG` and ignore arm joints
- Build differential drive bot from primitives
