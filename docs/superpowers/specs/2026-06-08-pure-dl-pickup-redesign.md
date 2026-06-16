# Design Spec — Pure-DL Warehouse Pickup Robot (CLIP + YOLO Removed)

**Date:** 2026-06-08
**Status:** Approved (brainstorm), pending implementation plan
**Supersedes:** text-conditioned / category-aware framing in `CLAUDE.md`, `docs/project/project_overview.md`, `docs/specs/environment.md`, `docs/project/timeline_terbaru.md`
**Related:** `docs/superpowers/specs/2026-06-01-arm-pickup-design.md` (earlier arm spec — this revises it without YOLO/CLIP)

---

## 1. Scope & Framing

**Old title:** "Text-Conditioned World Model for Visual Category-Aware Warehouse Robot" — Deep Learning + NLP + PCD, 5 people.

**New title:** **"Visual Goal-Conditioned World Model for Warehouse Pickup"** — pure Deep Learning.

### Removed
- **CLIP** (NLP component) — no text instructions, no `goal_emb`.
- **YOLO** (PCD/CV component) — no object detection. Box category/identity supplied directly in the observation, not perceived.
- Text-conditioned goals and the 512-dim language embedding pipeline.

### Kept / Added
- **DreamerV3** visual world model (the DL core).
- Isaac Lab 5.1 warehouse env, Ridgeback-Franka mobile manipulator.
- **Franka arm is now active** (was tucked) — full pick → carry → place task.
- 3 colored delivery zones, color/category conditioning via one-hot signal.

### Task
Spawn in receiving area → navigate to the **commanded** box → grasp it → carry to the **matching color zone** → place. Which box and which zone are both selected by a one-hot `goal_id`. No perception of category — identity is given.

---

## 2. Observation Contract (v2)

Interface contract for P2 (world model). One key removed (`goal_emb`), four manipulation keys added.

```python
obs = {
    # --- navigation ---
    "pixels":   Tensor(batch, 3, 64, 64),   # onboard RGB camera, float [0,1]
    "position": Tensor(batch, 3),            # robot base xyz, env-local
    "heading":  Tensor(batch, 2),            # [cos(yaw), sin(yaw)]
    "goal":     Tensor(batch, 3),            # delivery zone xyz, env-local — ANNEALS to zeros (curriculum)
    "goal_id":  Tensor(batch, 3),            # one-hot color/category — selects WHICH box + WHICH zone

    # --- manipulation ---
    "ee_pos":   Tensor(batch, 3),            # end-effector xyz, base frame (proprioception)
    "gripper":  Tensor(batch, 1),            # finger opening, 0 (closed) .. 1 (open)
    "holding":  Tensor(batch, 1),            # 1.0 if target box currently grasped, else 0.0
    "box_pos":  Tensor(batch, 3),            # target box xyz, env-local — UNANNEALED (grasp needs precision)
}
```

### Semantics
- `goal_id` is one-hot over `[orange, cyan, purple]`. It drives **both** the target box (size 21cm / 32cm / 52cm) and the delivery zone color. No detection step — the env tells the robot which box is the target.
- `goal` (delivery zone xyz) anneals to zeros across the curriculum so the policy learns to rely on `goal_id` + pixels for the *delivery* phase.
- `box_pos` is **always provided** (unannealed). Grasping requires sub-centimeter precision; annealing it would make the pick phase intractable for this project's scope.
- `ee_pos` is in the base frame so the policy always knows where its hand is relative to the chassis.
- EE orientation is fixed top-down (gripper points down) → quaternion is constant → not included in obs.

### P2 integration delta
- Remove the 512→64 projection that existed for CLIP. `goal_id` (3-dim) feeds the RSSM directly.
- New low-dim keys (`ee_pos`, `gripper`, `holding`, `box_pos`) concatenate into the existing low-dim observation vector.

---

## 3. Action Space (v2)

Flat continuous, shape **(6,)**, all components in `[-1, 1]`:

```python
action = [base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper]
```

| Component | Meaning | Mapping |
|---|---|---|
| `base_lin` | base linear velocity | × max_linear (1.5 m/s) → `_base_cmd` yaw projection |
| `base_ang` | base angular velocity | × max_angular (1.5 rad/s) → dummy revolute_z |
| `ee_dx/dy/dz` | end-effector position delta, base frame | scaled → target EE pose → DifferentialIKController → panda_joint1..7 |
| `gripper` | grasp command | > 0 → open, ≤ 0 → close (panda_finger_joint1/2) |

- EE orientation held fixed top-down by the IK controller; policy only commands EE position.
- One unified continuous policy (no mode switching) — DreamerV3-friendly.

### Arm kinematics — use existing tooling (do NOT hand-roll IK)
- **`isaaclab.controllers.DifferentialIKController`** — built-in differential inverse kinematics for Franka. Input: target EE pose. Output: joint position targets. This is the primary tool. [High]
- Reference implementations to copy: `Isaac-Reach-Franka-v0`, `Isaac-Lift-Cube-Franka-v0` (ship with Isaac Lab — working Franka arm control + reward). [High]
- cuRobo (NVIDIA GPU motion planner) is available if collision-aware planned reaches are later needed — out of scope for v1. [High]

---

## 4. Reward — Staged Pick-Place

Reward auto-switches between phases on the `holding` flag. Dense term follows the active phase.

```python
# Phase A — approach + grasp (while NOT holding)
-  0.01 * dist(ee_pos, box_pos)        # dense: pull hand toward target box
+  5.0  * grasp_success                # box gripped AND lifted off shelf > threshold

# Phase B — carry + place (while holding)
-  0.01 * dist(box_pos, goal_zone)     # dense: pull box toward correct color zone
+ 10.0  * delivery_success             # correct box in correct color zone, then released

# Always-on
-  0.005 * time_penalty                # per-step efficiency pressure
-  5.0   * collision                   # chassis contact force > 5N
-  2.0   * drop_penalty                # grasp lost mid-carry, box outside any zone
```

### Key points
- `grasp_success` (+5 intermediate) gives a bootstrap signal so the policy gets reward before completing the full chain — essential for DreamerV3 to learn pick-place. (Pure-sparse rejected as too hard for project scope.)
- `delivery_success` requires the category→color match encoded by `goal_id`. Delivering to the wrong zone yields no success reward.
- `drop_penalty` discourages losing the box during carry.

### 4b. Research methods retained — CA-SLOPE + Visual HER

Two methods from the original plan are **kept** (re-grounded for the no-CLIP/no-YOLO setting). They layer on top of the staged reward above; they do not replace it.

**Category-Aware SLOPE (CA-SLOPE)** — owner **P5** (reward method + RQ2 ablation).
- The dense shaping terms (`dist(ee_pos, box_pos)` in Phase A, `dist(box_pos, goal_zone)` in Phase B) become a **potential-based** landscape conditioned on the category from `goal_id`.
- Key change vs. original: category is read from `goal_id` (the commanded one-hot), **not** inferred from a visual detector (YOLO removed). The novelty stands — per-category potential landscapes — without the perception stage.
- RQ2: does conditioning the potential landscape on category (vs. one generic landscape) improve learning speed / success rate?

**Visual HER (Hindsight Experience Replay)** — owner **P3** (lives in the replay buffer).
- On a failed episode, relabel the achieved outcome (which box was actually grasped / which zone reached) as if it were the commanded `goal_id`, so failed rollouts still yield positive learning signal.
- Lives in P3's replay buffer (`buffer.add` / relabel on sample). Orthogonal to CA-SLOPE.

Both feed the same staged reward; CA-SLOPE reshapes the dense terms, Visual HER reuses off-goal experience.

---

## 5. Scene & Episode

| Item | Value | Change from before |
|---|---|---|
| Robot | Ridgeback-Franka, **arm active** | was arm tucked |
| Racks | 9 rack islands (3×3 grid) | unchanged — define the navigation maze |
| Boxes | **~18** graspable, one shelf level | was 54; loose boxes reduced |
| Box placement | bottom shelf (z≈0.72m), within Franka reach | NEW constraint — bottom rack level only (mid/top out of reach) |
| Delivery zones | 3 colored (orange / cyan / purple) | unchanged |
| Box categories | 21cm fragile→orange, 32cm regular→cyan, 52cm heavy→purple | unchanged sizes, no longer detected |
| Episode length | **1000 steps** (100s @ 10Hz) | was 600 (pick-place needs longer horizon) |
| Control freq | 10 Hz (decimation 20 @ 200Hz physics) | unchanged |
| Parallel envs | 1 (re-evaluate; fewer boxes may free VRAM) | unchanged target, possible headroom |

Box reachability is a hard constraint: any box that can be a target must spawn within the arm's workspace from a feasible base pose.

---

## 6. Team Roles (5 people, pure DL)

| Person | Old | New role |
|---|---|---|
| **P1 (Henry)** | Env & Integration | Env & Integration — Isaac Lab scene, obs/action/reward, **arm IK wiring** (DifferentialIKController) |
| **P2** | DreamerV3 | World model core — RSSM, encoder/decoder, world-model training |
| **P3** | ~~YOLOv8~~ | Policy — actor-critic, replay buffer, training loop, **Visual HER** (relabel in buffer) |
| **P4** | ~~CLIP~~ | Manipulation — grasp detection, pick-place curriculum, EE control tuning (pairs with P1) |
| **P5** | Coordinator | Experiments — **CA-SLOPE** reward method + RQ2 ablation, eval metrics, baselines (SAC/PPO), W&B, paper |

---

## 7. Curriculum

Phased difficulty (DreamerV3 trained throughout):

1. **Nav only** — box pre-grasped (`holding=1` at spawn), learn carry → place. Isolates delivery.
2. **Grasp only** — robot spawns at the box, learn approach → grasp. Isolates manipulation.
3. **Full chain** — spawn in receiving area, navigate → grasp → carry → place. `goal` xyz still provided.
4. **Anneal goal** — `goal` (delivery zone xyz) anneals to zeros; policy relies on `goal_id` + pixels for delivery. `box_pos` stays provided.

---

## 8. Out of Scope (v1)

- Any perception/detection of box category (given directly via `goal_id`; CA-SLOPE reads category from `goal_id`, not vision).
- Text / language conditioning (CLIP removed).
- 6-DOF dexterous EE orientation control (fixed top-down).
- cuRobo motion planning (DifferentialIKController only).
- Multi-box / sequential delivery in one episode (single box per episode).
- `box_pos` annealing / vision-only grasp.

(CA-SLOPE and Visual HER are IN scope — see §4b.)

---

## 9. Open Items for Implementation Plan

- Verify Franka workspace vs. chosen shelf height (explore script) — confirm ~18 box positions are reachable.
- Grasp-success detection: finger contact + box lift threshold — define exact criteria.
- VRAM re-measure with 18 boxes + active arm IK — confirm 1 env (or more) fits RTX 5050 8GB.
- DifferentialIKController config: base-frame vs. world-frame EE target convention.
- Action scaling constants for `ee_dx/dy/dz` (reach per control step).
