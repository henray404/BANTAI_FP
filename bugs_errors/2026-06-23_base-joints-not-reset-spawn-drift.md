# Base dummy joints not reset on spawn → crashed/bounds at ep_len≈5

**Date:** 2026-06-23
**Severity:** CRITICAL (training unusable — every episode dies in ~5 steps)
**Status:** FIXED (code; needs sim confirmation)

## Symptom
Training log: `train_length 5.0` constant, `train_episodes` climbing fast, `train_return ≈ -1.6`.
Per-step DIAG-term probe:
```
[DIAG-term] ep_len=0 fired=['crashed'] contactN=545261184 base_xy=(-4.27,18.71) ...
[DIAG-term] ep_len=0 fired=['bounds']  contactN=0        base_xy=(0.53,23.22)  ...
[DIAG-term] ep_len=0 fired=['bounds']  contactN=0        base_xy=(17.80,15.30) ...
```
`base_xy` at reset is far OUTSIDE the arena (|x|>9.5, |y|>14.5) even though the spawn range is
x∈[-8,8], y∈[10,12.5]. Contact force spikes to 1e3–1e8 N. Nondeterministic across runs.

## Root cause
The Ridgeback-Franka is a fixed-root articulation: `base_link` = root_anchor + dummy-base-joint
offset (`dummy_base_prismatic_x/y_joint`, `dummy_base_revolute_z_joint`). Those base joints are
**velocity-controlled**, so their integrated POSITION is never commanded back to zero.

`EventCfg.reset_robot` uses `reset_root_state_uniform`, which writes **only the root anchor**. There
was **no joint-reset event** and `_reset_idx` zeroed the base joints **only** in the stage-2
`_spawn_base_near_box` path. On a normal (STAGE_FULL/default) reset the base joints kept the previous
episode's drift, so `base_link` = north_anchor + stale_offset → spawned out of bounds or penetrating
a rack → `crashed`/`bounds` fired right after the RESET_GRACE_STEPS=5 window.

Reward-independent — this is why reward retuning (incl. the PBS migration) had zero effect on it, and
why stage-2 runs (which zero the joints) behaved differently.

## Fix
`env/warehouse_env.py`: extracted `_zero_base_joints(env_ids)` and call it in `_reset_idx` after
`super()._reset_idx()` for ALL stages (stage-2 still re-zeros after moving the anchor — idempotent).

## Verify (sim)
Re-run training; the DIAG-term `base_xy` at reset must now be inside x∈[-8,8], y∈[10,12.5], episodes
should survive past 5 steps, `train_length` should climb. Remove the temp DIAG-term probe once green.
