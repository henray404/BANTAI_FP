# Mission Specification — Warehouse Pickup Task

> Pure-DL pickup task (NLP/PCD dropped 2026-06-08). Full redesign spec:
> `docs/superpowers/specs/2026-06-08-pure-dl-pickup-redesign.md`.

## One-line mission
Spawn at receiving-north → navigate to the commanded box → grasp it → carry it to the
matching color zone → release. **One box per episode.**

## Task type
Pick → carry → deliver, goal-conditioned. Both the target box AND the delivery zone are
selected by a single `goal_id` one-hot. **No perception of category** — the box category is
given directly (CLIP text + YOLO detection removed 2026-06-08). The learning challenge is
navigation + grasp + carry + place, not detection.

## How a task is generated (per episode, `_sample_targets`)
1. Draw a random category `c ∈ {0=fragile, 1=regular, 2=heavy}`.
2. Pick a random box of category `c` → `target_box_name`.
3. Set `goal_pos` = the zone matching `c` (zone order == category order).
4. Set `goal_id` = one-hot of `c`.
Then `_randomize_box_poses` jitters box x,y and `_refresh_target_box_pos` writes `box_pos`.

## What the robot is told (obs) vs must do
| Given via obs | Robot must achieve |
|---|---|
| `goal_id` — which category (→ which zone) | drive to the box, grasp it |
| `box_pos` — exact target box xyz (unannealed) | lift box off the floor |
| `goal` — delivery zone xyz (annealed in curriculum) | carry to matching color zone |
| `pixels`, `position`, `heading`, `ee_pos`, `gripper`, `holding` | release inside zone |

> `box_pos` is given precisely, so the robot does not *search* for the box. `goal_id`'s
> research value (category-aware behavior) lives in the **delivery** stage and in Visual HER.

## Success / failure
| Event | Condition | Reward |
|---|---|---|
| `grasp_success` | fingers closed on box + box lifted off floor → sets `holding=1` | +5 (one-shot) |
| `delivery_success` | held box of the commanded category inside its matching color zone, then released | +10 |
| `drop_penalty` | box dropped mid-carry outside any zone | −2 (one-shot) |
| `collision` | chassis contact > 5 N | −5 per step |
| Out of bounds | `|x|>9.5` or `|y|>14.5` | episode ends |

Delivery radius: box within **1.5 m** of the commanded zone center counts as delivered.

## Curriculum (spec §7)
nav-only (pre-grasped box) → grasp-only → full chain → anneal `goal` (zone xyz → zeros, so the
policy must use `goal_id` + vision). `box_pos` stays unannealed throughout (grasp needs precision).

## Episode parameters
| Param | Value |
|---|---|
| Horizon | 1000 steps (100 s × 10 Hz) |
| Spawn | receiving-north `x[-8,8] y[11,14]`, yaw random |
| Parallel envs | 1 |

## Research hooks (not P1's code, but mission-relevant)
- **Visual HER** (P3) — relabels failed episodes by the box/zone actually reached, so a wrong-box
  rollout still gives positive signal. See `buffer/visual_her.py`.
- **CA-SLOPE** (P5) — category-aware reward reading category from `goal_id` (not vision); RQ2 ablation.

## Method
Visual goal-conditioned **world model** (DreamerV3): P2 RSSM world model → P3 actor-critic trained
in imagination, fed the obs above + `goal_id`. Baselines: SAC/PPO (P5).
