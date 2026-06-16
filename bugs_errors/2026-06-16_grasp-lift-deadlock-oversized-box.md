# Grasp never succeeds — lift deadlock + center-distance unreachable for oversized boxes

**Date:** 2026-06-16
**Component:** `env/grasp.py` (`grasp_success` / `grasp_lost`), `env/warehouse_env.py` (`update_grasp`)
**Severity:** High — `holding` can never become 1, so grasp/carry/delivery cannot be learned

## Symptom
`holding` stays 0 in teleop even with the gripper closed right on a target box.

## Root cause
`grasp_success(ee_pos, box_pos, gripper_closed, box_lift)` required ALL of:
1. `gripper_closed` (finger < 0.0175)
2. `near = ||ee_pos - box_pos|| < GRIP_RADIUS_M (0.08)` — distance to box **center**
3. `lifted = box_lift > LIFT_M (0.05)`

Three problems, all rooted in boxes being far larger than the Franka gripper (opening ~8 cm;
boxes are 0.21 / 0.32 / 0.52 m):

- **Enclosure impossible** — the gripper cannot wrap a box bigger than its opening. "Closed"
  just means fingers shut against the surface.
- **Center distance unreachable** — `near` measures to the box *center*. A 0.52 m box has its
  center 0.26 m inside; the EE cannot get within 0.08 m of the center without penetrating the box.
  So `near` is never true for regular/heavy boxes.
- **Lift deadlock** — `box_lift` only rises once the box is carried, but carry
  (`_carry_held_boxes`) only runs after `holding=1`, which needs `lifted` first. Chicken-and-egg;
  the box never lifts via physics (too big/heavy for the gripper), so grasp never fires.

## Resolution
Switch to a contact/proximity attach model (the kinematic carry in `_carry_held_boxes` already
exists — only detection was broken):
- Measure distance to the box **surface** (`||ee-box|| - box_half`), size-aware via `box_half`.
- Grasp = `gripper_closed AND surface_dist < GRIP_RADIUS_M`. **Drop the lift precondition** — lift
  becomes a consequence of the kinematic carry, not a prerequisite.
- `grasp_lost` uses the same surface distance.

`update_grasp` passes per-env `box_half` (= target box size / 2) from a new `self._box_size` map.
The robot must still command the gripper closed near the box (policy learns approach + close),
but is no longer required to physically enclose/lift an oversized box.
