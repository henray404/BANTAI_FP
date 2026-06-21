# Stage-2 spawn lands the base INSIDE the rack (physics blowup)

**Date:** 2026-06-21
**Area:** env/warehouse_env.py `_spawn_base_near_box` (curriculum stage 2 / demo_pickup default)
**Severity:** high — demo unusable, robot spawns interpenetrating the rack and explodes

## Symptom
Running `scripts/demo_pickup.py` (stage 2), the robot spawns embedded in the rack island and the
sim bugs out (interpenetration -> the solver flings the base). Reported by Henry.

## Root cause
`_spawn_base_near_box` placed the base only `standoff` = 0.55 m (heavy) / 0.65 m (small) north of
the box, and the box sits at the rack centre. The rack footprint is ~0.9 m deep (shelf decks are
0.70 m, `SHELF_DECK_SIZE`), so its north face is ~0.35-0.45 m from the box centre. The Ridgeback
base is ~0.96 m long (~0.5 m half-length), so a base CENTRE at 0.55-0.65 m puts the base's front
half (and the Franka column) inside the rack frame / shelf-deck colliders -> interpenetration.

The old standoff was deliberately tuned small (2026-06-20) so the FROZEN hand auto-grabs on spawn
(magnetic pickup). That optimisation is what drove the base into the rack.

## Fix
Spawn the base in the OPEN AISLE clear of the rack (`standoff` 1.0 m small / 1.1 m heavy =
deck-half ~0.35 + robot-half ~0.5 + margin), and let the demo controller / policy drive the last
~0.4 m in so the magnetic grasp latches on proximity. No more spawn interpenetration; the brief
approach is harmless (and more realistic) for stage-2 grasp isolation.

## Verify
`python scripts/demo_pickup.py` — robot spawns in the aisle facing the box (NOT inside the rack),
drives forward a short way, grabs, carries to the zone.
Tune `standoff` here if the base still clips (raise) or can't reach grasp range (lower).
