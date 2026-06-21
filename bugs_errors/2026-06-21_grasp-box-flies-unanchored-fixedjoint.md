# Grasp box flies away — FixedJoint has no local anchor frames

**Date:** 2026-06-21
**Area:** env/attach.py, env/warehouse_env.py (CARRY_MODE="physics")
**Severity:** high — pickup demo unusable (box visibly flung off the robot)

## Symptom
On grasp the box does NOT stay at the hand. It is flung far away, but its motion direction
tracks the robot (it moves the same way the chassis moves). Reported by Henry running
`scripts/demo_pickup.py`.

## Root cause
`attach_box()` defined a `UsdPhysics.FixedJoint` and set Body0Rel (base_link) + Body1Rel (box),
but never authored `localPos0/localRot0/localPos1/localRot1`. With the anchor frames unset they
default to identity on BOTH bodies, so PhysX solves the constraint to make the base_link origin
coincide with the box origin. The box was snapped to the carry anchor (~0.6 m fwd, 0.7 m up)
before the weld, so on the first physics step the solver violently pulls the box to satisfy the
(identity==identity) constraint -> "terbang jauh", then the welded box rides with the chassis ->
"arah gerak sama kayak robot".

The misleading comment in attach.py ("PhysX freezes the body0->body1 relative transform at
definition time") is false for UsdPhysics — the relative pose must be authored via the local
anchor attrs; it is not auto-captured from current world transforms.

## Fix
1. `attach_box()` now accepts `local_pos0` / `local_rot0` and authors all four anchor attrs
   (box anchored at its own origin: localPos1=0, localRot1=identity).
2. `WarehouseRLEnv._attach_boxes()` computes the box pose in the base_link frame at weld time via
   `isaaclab.utils.math.subtract_frame_transforms(base_pos, base_quat, box_pos, box_quat)` and
   passes it as the body0 anchor, so the box holds exactly at the carry anchor.
3. `_snap_boxes_to_ee()` now also resets box orientation to upright (qw=1) so the welded relative
   rotation is deterministic.

## Verify
`python scripts/demo_pickup.py` — box stays ~0.6 m in front of + above the chassis through the
whole carry, no fling, delivered into the zone.
