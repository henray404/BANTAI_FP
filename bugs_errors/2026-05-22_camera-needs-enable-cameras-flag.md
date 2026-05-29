# Bug: Camera sensor spawn fails without --enable_cameras

**Date:** 2026-05-22
**File:** scripts/run_env.py, tests/test_env.py, tests/test_obs.py
**Status:** [x] Fixed

---

## Error Message
```
RuntimeError: A camera was spawned without the --enable_cameras flag. Please use --enable_cameras to enable rendering.
  at c:/isaaclab/source/isaaclab/isaaclab/sensors/camera/camera.py:396 in _initialize_impl
```

## What I Was Doing
Running `python scripts\run_env.py --num_envs 1` for first visual sanity test.
Sim launched, articulation spawned, then crashed when the camera sensor tried
to initialize.

## Root Cause
Isaac Lab's `Camera._initialize_impl` requires the carb setting that
AppLauncher only enables when given `--enable_cameras`. Without the flag,
camera-bearing scenes cannot launch.

## Fix
Force `args_cli.enable_cameras = True` in every entry script before calling
`AppLauncher(args_cli)`. The warehouse env always has an onboard camera, so
the flag should not be a user concern.

```python
args_cli = parser.parse_args()
args_cli.enable_cameras = True   # warehouse env always uses the onboard cam
app_launcher = AppLauncher(args_cli)
```

Applied in: `scripts/run_env.py`, `tests/test_env.py`, `tests/test_obs.py`.
