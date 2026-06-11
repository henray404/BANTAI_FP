# Bug: Onboard camera frames identical while robot drives (camera mounted on frozen root)

**Date:** 2026-06-09
**File:** env/warehouse_scene.py (WarehouseSceneCfg.camera); affects obs["pixels"], scripts/drive_env.py, tests/test_env.py
**Status:** [x] Fixed (2026-06-09) — re-parent camera under base_link.

---

## Symptom

Driving the robot in `scripts/drive_env.py --cam N` produced identical camera PNGs — the onboard
view never changed even though the chassis translated/rotated. obs["pixels"] is effectively static.

## Root Cause

The camera prim_path was `{ENV_REGEX_NS}/Robot/onboard_cam` — parented directly under `/Robot`,
whose root link is the welded `world` link. That root stays FIXED at the env origin (IsaacLab
#1268: the dummy holonomic joints translate `base_link`, not the articulation root). So the camera,
pinned to the static root, never moved with the chassis → every frame identical.

Same root-frozen family as `bugs_errors/2026-06-09_ridgeback-floating-base.md` and
`2026-06-04_ridgeback-root-state-frozen.md` (obs read the fixed root): anything attached to the
articulation root does not move; it must hang off `base_link`.

## Fix

Re-parent the camera under the moving chassis body:

```python
# before
prim_path="{ENV_REGEX_NS}/Robot/onboard_cam"
# after
prim_path="{ENV_REGEX_NS}/Robot/base_link/onboard_cam"
```

`/Robot/base_link` is a valid prim (the contact sensor already attaches there, warehouse_scene.py
:445). OffsetCfg pos (0.35, 0, 0.55) / ros convention is unchanged — now applied relative to
base_link, so the camera rides the chassis.

## Verify

`python scripts/drive_env.py --cam 20` → drive forward / yaw → consecutive `_cam_debug/cam_*.png`
must now DIFFER and track the robot's heading.

## Note (separate, cosmetic)

The MDL `C120 could not find module ... Materials::Base::...` errors seen at load are unrelated:
the rack USDs reference NVIDIA core materials by relative path that don't resolve, so racks render
untextured. Not fatal — physics, camera, and obs are unaffected. Track separately if textures are
wanted.
