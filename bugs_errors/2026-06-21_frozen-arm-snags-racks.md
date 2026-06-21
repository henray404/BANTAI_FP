# Frozen Franka arm snags racks during carry (robot stuck)

**Date:** 2026-06-21
**Area:** env/warehouse_scene.py arm pose + env/warehouse_env.py (magnetic-pickup carry)
**Severity:** high — robot can't carry box to zone, base jams when the arm catches a rack

## Symptom
After the spawn/avoidance/spin fixes, the base drives but gets stuck again — the Franka arm
catches on a rack frame mid-carry and the base can't move past. Reported by Henry.

## Root cause
The arm is FROZEN in the Franka "ready" pose (panda_joint2=-0.569, joint4=-2.810, joint6=2.0 ->
forearm + hand extend forward). In the magnetic-pickup design the arm never actuates: the grasp is
a proximity check and the carried box is welded to base_link, so the arm is non-functional
scenery. But its colliders are live, so the extended arm physically catches rack frames while the
base drives past them -> base contact constraint stops the chassis -> stuck.

## Fix
`WarehouseRLEnv._disable_arm_collisions()` (called once in __init__) turns off `UsdPhysics`
collision on every prim under the robot whose path contains `/panda_` (arm links + hand + fingers
+ their collision meshes). Base/chassis colliders are untouched, so the base still blocks against
racks and the potential-field avoidance still works. The arm now passes through racks (harmless
visual — it carries nothing; the box rides on base_link).

## Verify
`python scripts/demo_pickup.py --config configs/demo_tuning.yaml --episodes 1 --log_csv demo_loc.csv`
— base carries the box to the zone without jamming; `move` stays > 0 through the carry phase.

## Alternative (not taken)
Tuck the arm to a compact upright pose instead — but then the magnetic grasp (proximity of the
hand to the box) needs a much larger GRIP_RADIUS_M, making the grab look like the box teleports
from far. Disabling arm collision keeps the natural reach-into-shelf grab.
