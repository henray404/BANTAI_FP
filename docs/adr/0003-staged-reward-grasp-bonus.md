# ADR 0003 — Staged Pick-Place Reward with Grasp Bonus

- **Status:** Accepted
- **Date:** 2026-06-08
- **Spec:** `docs/superpowers/specs/2026-06-08-pure-dl-pickup-redesign.md` §4, §4b

## Context

The pick → carry → place task has a long horizon (1000 steps). A pure-sparse reward (only +reward on final delivery) gives the policy almost no signal until it accidentally completes the entire chain — intractable for DreamerV3 within this project's compute and timeline. The task also has two distinct sub-goals (grasp, then deliver) with different dense gradients.

## Decision

Staged reward that auto-switches phase on the `holding` flag, with an intermediate grasp bonus.

```
Phase A (not holding):  -0.01·dist(ee_pos, box_pos)   +5.0·grasp_success
Phase B (holding):      -0.01·dist(box_pos, goal_zone) +10.0·delivery_success
Always-on:              -0.005·time  -5.0·collision  -2.0·drop_penalty
```

- **+5.0 grasp_success** is the key bootstrap: the policy earns reward on a partial milestone (box gripped AND lifted off shelf) before the full chain. Pure-sparse was explicitly rejected.
- **delivery_success** requires the category→color match from `goal_id`; wrong zone yields no success reward.
- Dense term follows the active phase only.

### Research methods layered on top (do not replace staged reward)

- **CA-SLOPE** (P5): the dense shaping terms become a potential-based landscape conditioned on category, read from `goal_id` (not vision). RQ2: per-category landscapes vs. one generic — learning speed / success rate.
- **Visual HER** (P3): relabel a failed episode's achieved outcome as the commanded `goal_id` so off-goal rollouts still teach. Lives in the replay buffer.

## Consequences

- Episode length raised to **1000 steps** (was 600) for the longer horizon.
- Boxes reduced to **~18** (from 54), one shelf level within Franka reach, so the pick phase is feasible.
- Reward code must read `holding` to select the active phase and dense term.
- CA-SLOPE and Visual HER are in scope; both feed this same staged reward.
