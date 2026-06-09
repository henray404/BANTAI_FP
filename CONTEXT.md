# CONTEXT — Domain Glossary

Canonical vocabulary for the Warehouse Pickup project. Use these terms exactly in issue titles, ADRs, test names, and code. Don't drift to synonyms. If a concept you need isn't here, it's a signal: either you're inventing language the project doesn't use, or there's a real gap to record via `/grill-with-docs`.

Project: **Visual Goal-Conditioned World Model for Warehouse Pickup** — pure Deep Learning (DreamerV3). NLP (CLIP) and PCD (YOLO) dropped 2026-06-08.

---

## Core nouns

| Term | Definition |
|---|---|
| **Task** | One episode: spawn in receiving area → navigate to the commanded box → grasp → carry to the matching color zone → place. Single box per episode. |
| **Robot** | **Ridgeback-Franka** — Clearpath holonomic base (3 dummy base joints) + Franka Panda 7-DOF arm + parallel gripper. Replaced Carter/Jetbot 2026-06-01. |
| **Base** | Holonomic chassis. Driven via `_base_cmd` mapping `(base_lin, base_ang)` → `[vx, vy=0, wz]` on dummy prismatic/revolute joints. No wheel kinematics. |
| **Arm** | Franka Panda 7-DOF, **active** (was tucked). Position-controlled via `DifferentialIKController`. EE orientation fixed top-down. |
| **EE** | End-effector (gripper tip). Commanded by position delta `(ee_dx, ee_dy, ee_dz)` in **base frame**. |
| **Box** | Graspable target object. ~18 in scene, one shelf level, within Franka reach (~0.85m). Size encodes category. |
| **Rack island** | One of 9 shelving units in a 3×3 grid. Define the navigation maze. |
| **Delivery zone** | One of 3 colored floor regions: orange / cyan / purple. Target zone selected by `goal_id`. |
| **Receiving area** | North spawn region (x:−8..+8, y:+11..+14, yaw random). |

## Category ↔ color ↔ size

The single most important mapping. Category is **given** via `goal_id`, never detected (YOLO removed).

| `goal_id` one-hot | Color | Box size | Category label |
|---|---|---|---|
| `[1,0,0]` | orange | 21 cm | fragile |
| `[0,1,0]` | cyan | 32 cm | regular |
| `[0,0,1]` | purple | 52 cm | heavy |

`goal_id` selects **both** the target box AND the matching delivery zone.

## Observation keys

The obs dict is an interface contract (P2 world model consumes it). Do NOT rename or reshape without team discussion.

| Key | Shape | Meaning | Anneal? |
|---|---|---|---|
| `pixels` | (B,3,64,64) | onboard RGB, float [0,1] | — |
| `position` | (B,3) | base xyz, env-local | — |
| `heading` | (B,2) | `[cos(yaw), sin(yaw)]` | — |
| `goal` | (B,3) | delivery zone xyz | **anneals to zeros** (curriculum) |
| `goal_id` | (B,3) | one-hot category/color | — |
| `ee_pos` | (B,3) | EE xyz, base frame | — |
| `gripper` | (B,1) | finger opening 0..1 | — |
| `holding` | (B,1) | 1.0 if target box grasped | — |
| `box_pos` | (B,3) | target box xyz, env-local | **unannealed** (grasp needs precision) |

## Action

`action = [base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper]`, shape (6,), all in [−1, 1]. `gripper > 0` open, `≤ 0` close.

## Reward / termination terms

| Term | Phase | Value |
|---|---|---|
| **grasp_success** | A (not holding) | +5.0 — box gripped AND lifted off shelf > threshold. Intermediate bootstrap signal. |
| **delivery_success** | B (holding) | +10.0 — correct box in correct color zone, then released. |
| **dist shaping** | active phase | −0.01 × distance (Phase A: ee→box; Phase B: box→zone). |
| **collision** | always | −5.0 on chassis contact force > 5N. |
| **drop_penalty** | always | −2.0 — grasp lost mid-carry, box outside any zone. |
| **time_penalty** | always | −0.005 per step. |
| **holding** | — | flag that switches the active reward phase A↔B. |

## Phases

- **Phase A** — approach + grasp (while NOT holding).
- **Phase B** — carry + place (while holding).

## Curriculum stages

1. **Nav only** — box pre-grasped at spawn, learn carry→place.
2. **Grasp only** — spawn at box, learn approach→grasp.
3. **Full chain** — receiving → navigate → grasp → carry → place; `goal` xyz still provided.
4. **Anneal goal** — `goal` anneals to zeros; rely on `goal_id` + pixels for delivery. `box_pos` stays provided.

## Research methods (retained, no CLIP/YOLO)

| Method | Owner | One line |
|---|---|---|
| **CA-SLOPE** (Category-Aware SLOPE) | P5 | Potential-based dense shaping conditioned on category read from `goal_id` (not vision). RQ2 ablation. |
| **Visual HER** (Hindsight Experience Replay) | P3 | Relabel failed episode's achieved outcome as the commanded `goal_id` so off-goal rollouts still teach. Lives in replay buffer. |

## Stack terms

| Term | Meaning |
|---|---|
| **DreamerV3** | The DL core — visual world model (RSSM + actor-critic in imagination). |
| **RSSM** | Recurrent State-Space Model — DreamerV3's latent dynamics. |
| **DifferentialIKController** | `isaaclab.controllers` built-in differential IK. Target EE pose → Franka joint targets. Do NOT hand-roll IK. |
| **Isaac Lab** | Simulator, v5.1. Conda env `isaaclab`, path `C:\IsaacLab`. |
