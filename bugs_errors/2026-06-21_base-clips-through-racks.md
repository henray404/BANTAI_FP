# Base clips through racks (robot tembus rak)

**Date:** 2026-06-21 · **Area:** env physics (base drive vs static rack colliders)
**Status:** FIX APPLIED — **UNVERIFIED on hardware** (Blackwell camera blocker; verify on the
Linux 2-GPU box per `docs/setup/TRAINING_2GPU.md`).

## Symptom
During policy rollouts the Ridgeback base drives straight through rack islands instead of
being stopped by them — no physical collision response against the racks.

## Root-cause hypotheses (ranked)
1. **Over-strong base velocity drive (primary).** `RIDGEBACK_FRANKA_CFG` base actuator shipped
   `effort_limit_sim=100000.0`, `damping=1e5` on the holonomic dummy joints. A velocity drive
   that strong applies enough force to satisfy the commanded velocity that it overpowers the
   rack contact constraint before the solver can depenetrate — the base bulldozes through. This
   is the classic "kinematic-ish velocity base tunnels through static obstacles".
2. **Rack collision approximation.** Racks are static `AssetBaseCfg(... CollisionPropertiesCfg())`
   loaded from `Rack_L01...usd` (scaled 0.01). If the USD authors no/poor colliders for the open
   frame, the base only collides with the shelf-deck cuboids (which DO have collision — boxes land
   on them), not the uprights. Open shelving has lots of empty space, so partial clipping looks
   like full pass-through.
3. **No CCD / depenetration head-room.** At 1.5 m/s with 200 Hz physics, per-substep motion is
   small so pure tunneling is unlikely — but combined with (1) the solver never wins.

## Fix applied
`env/warehouse_scene.py`:
- New constant `BASE_DRIVE_EFFORT = 2000.0`; base actuator now uses it (was `100000.0`),
  `damping` `1e5 -> 1e4`. Force budget is still ample to drive a ~100 kg base at 1.5 m/s but
  low enough that a static rack contact constraint wins.

This addresses hypothesis (1), the most likely cause, and is fully reversible (one constant).

## Verify when the sim runs (Blackwell-blocked now)
1. `python scripts/drive_robot.py` — manually drive the base into a rack island. Expect it to
   STOP, not pass through.
2. If it still clips: lower `BASE_DRIVE_EFFORT` (e.g. 1000), and inspect the rack USD colliders
   via `python asset_sandbox/scripts/explore_rack.py` — if the frame has no collider, add an
   invisible box collider around each rack footprint in `_rack_cfg`.
3. If the base feels sluggish / can't push past floor friction: raise `BASE_DRIVE_EFFORT`.
4. Cross-check with the existing collision penalty + contact sensor (`collision_penalty` reward)
   — a working physical block should also reduce the rate of collision-penalty events.

## Notes
- The contact sensor (`activate_contact_sensors=True`) already exists for the reward-level
  collision penalty; that is independent of the physical blocking fixed here.
- Do NOT raise `effort_limit_sim` back to 1e5 — that reintroduces this bug.
