# Bug: Stage-2 spawn standoff too tight → base clips rack → `crashed` at ep_len≈8

**Date:** 2026-06-24
**File:** env/warehouse_env.py (`_spawn_base_near_box`), env/curriculum.py (`spawn_pose_near_box`)
**Status:** [x] Fixed (raised standoff) — verify in sim

---

## Symptom
After the GPU-solver crash was fixed (force CPU physics, commit e05e439), training ran but every
stage-2 episode died in ~5-14 steps:
```
[DIAG-term] ep_len=0 fired=['crashed'] contactN=3764 base_xy=(-5.26,8.98) holding=False ...
[DIAG-term] ep_len=0 fired=['crashed'] contactN=3949 base_xy=(-6.96,8.85) holding=False ...
[NNNN] dataset_size ... / train_length 8.0 / train_episodes ...   # lengths collapse 430 -> 8
```
`success_rate` stuck at 0, robot just moves back-and-forth (degenerate). `crashed` fires at
chassis contact > 50 N (`warehouse_env.py:475`), but contactN was in the THOUSANDS at spawn.

## Root cause
`_spawn_base_near_box` places the chassis `standoff` m from the box along a FIXED north
(`approach_dir=(0,1)`) vector. The direction is fine (every island row's aisle opens north,
rows are 6-7 m apart), but the **standoff was too short**:

- rack/shelf-deck footprint ~0.9 m deep → half-depth 0.45 m
- Ridgeback base ~0.96 m long → half-length 0.48 m
- edges touch at center-to-center = 0.45 + 0.48 = **0.93 m**

Old standoff 1.0 (normal) / 1.1 (heavy) left only **~0.07-0.17 m** between the base front and the
rack's own north face. Box never sits exactly at center + spawn jitter + the lowered
`max_depenetration_velocity=0.5` (slower overlap resolution) → the base spawned slightly INSIDE
the rack face → contact force spiked to thousands of N → `crashed` reset before the policy could
move. Worthless episodes → no learning → back-and-forth attractor (also amplified by
collision/under_rack penalties making "crash fast" cheaper than "stay alive"; see reward notes).

## Fix
Raise the stage-2 standoff to clear the footprint with ~0.3 m margin:
`standoff = 1.4 if heavy else 1.25` (was `1.1 / 1.0`). The policy/demo still drives the last
~0.6 m in for the magnetic grab. Per-box `_spawn_standoff` override path unchanged.

## Not fixed here (separate)
- `grasp_joint disjointed body transforms ... snap objects together` warnings: when the box is
  FixedJoint-welded to the hand the bodies aren't perfectly coincident → snap → contact spike →
  some `crashed` with holding=True (contactN~7000). Grasp-attach geometry, P4 territory.
- Reward landscape penalty-dominated (collision/under_rack/idle all negative; grasp/deliver rarely
  trigger) — tune with P5/jere once approach is reliable.

## Related
- bugs_errors/2026-06-21_stage2-spawn-inside-rack.md (same family: standoff < 0.85 spawned INSIDE rack)
- bugs_errors/2026-06-23_oom-bar1-physx-error700.md (the GPU-solver crash fixed just before this surfaced)
