# Design Spec — Category-Aware Box Pickup with Mobile Manipulator

**Date:** 2026-06-01
**Author:** P1 (Henry)
**Status:** Approved (brainstorming) — pending implementation plan
**Supersedes pickup intent in:** `CLAUDE.md` ("no pickup, Phase 1 only")

---

## 1. Goal

Extend the warehouse task from single-goal navigation to a full
**pick → carry → deliver** loop that exercises the project's "Visual Category-Aware"
thesis: an instruction names a box category; the robot must navigate to the
**correct-category** box, pick it with a robot arm, and deliver it to the **matching**
delivery zone. Wrong box or wrong zone is penalized.

Realism requirement (user): the robot has a **physical arm** (mobile manipulator), not a
fake-grab or flying base.

## 2. Scope & Non-Goals

**In scope**
- Mobile-manipulator robot (Ridgeback base + Franka Panda arm) from existing assets.
- 3 pickup stations (one per category), floor markers + a pickable box per station.
- A robot-agnostic pickup **state machine** (`SEEKING → REACHING → CARRYING → DONE`).
- **Scripted** arm reach + grasp (auto-IK), **kinematic attach** of box to gripper.
- Obs extension: `carrying` flag; `goal` switches marker → zone by phase.
- Reward + termination terms for pickup, delivery, and the two mistake cases.

**Non-Goals (explicit)**
- No **learned** arm/grasp control. The arm motion is deterministic; the policy only
  learns base navigation. (Keeps action space `(2,)`.)
- No **physics-friction** grasp (unstable, slips, poor RL signal). Grasp = kinematic attach.
- No holonomic base control. Holonomic Ridgeback is driven diff-drive-style to preserve
  the `(2,)` action contract.
- No curriculum changes here (goal→zeros vision annealing remains a later, separate spec).
- No CLIP/YOLO wiring changes; `goal_emb` stays as P4 owns it.

## 3. Interface Contract Changes (REQUIRES P2 awareness)

The observation contract in `CLAUDE.md` gains **one key**. Action space is **unchanged**.

```python
obs = {
    "pixels":   Tensor(batch, 3, 64, 64),   # unchanged — base-front camera
    "position": Tensor(batch, 3),            # unchanged — base xyz, env-local
    "goal":     Tensor(batch, 3),            # MEANING CHANGES: active target.
                                             #   phase SEEKING  -> correct marker xyz
                                             #   phase CARRYING -> matching zone xyz
    "goal_emb": Tensor(batch, 512),          # unchanged — CLIP (P4)
    "heading":  Tensor(batch, 2),            # unchanged — [cos,sin] yaw
    "carrying": Tensor(batch, 1),            # NEW — 0.0 not carrying / 1.0 carrying box
}

action = [linear_vel, angular_vel]           # UNCHANGED shape (2,), values [-1,1]
```

- `carrying` lets the world model distinguish the two task phases explicitly (DreamerV3
  can also see the box in `pixels`, but the flag is a robust, cheap signal).
- `goal` is already documented as "will anneal to zeros in curriculum" — its phase-dependent
  value is consistent with that plan.

## 4. Robot — Ridgeback-Franka (existing asset, ~zero build)

- **USD:** `{ISAAC_NUCLEUS_DIR}/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd`
  (Isaac 5.1 Nucleus, HTTP 200 verified). One pre-rigged articulation.
- **Config:** reuse Isaac Lab `RIDGEBACK_FRANKA_PANDA_CFG`
  (`isaaclab_assets/robots/ridgeback_franka.py`), `.replace(prim_path=...)`.
- **Joints (known from shipped cfg):**
  - Base (velocity ctrl, holonomic): `dummy_base_prismatic_x_joint`,
    `dummy_base_prismatic_y_joint`, `dummy_base_revolute_z_joint`.
  - Arm (position ctrl): `panda_joint1..7`.
  - Gripper (position ctrl): `panda_finger_joint1`, `panda_finger_joint2`.
- **Base drive (preserve `(2,)`):** map `[linear, angular]` →
  `prismatic_x` velocity = linear, `revolute_z` velocity = angular, `prismatic_y` = 0
  (no strafe). Ridgeback is holonomic but is intentionally constrained to diff-drive
  behaviour so the action contract is unchanged.
- **VRAM note:** heavier than Carter (10 DOF + arm bodies). Train headless, 1–2 envs on the
  8 GB RTX 5050. Confirm during smoke test; reduce to `num_envs=1` if needed.
- **Camera:** `TiledCameraCfg` mounted on a base-front link (keeps `pixels` semantics).
  Verify the mount link name from the articulation-init log on first run.

## 5. Scene — Pickup Stations (`warehouse_scene.py`)

Three stations, one per category. Each station = a **floor marker** + a **pickable box**
(distinct from the 18 decoration boxes, which remain for YOLO).

| Station | Category | Box size | Marker color | Matching zone |
|---|---|---|---|---|
| `st_A` | fragile | 0.21 m | orange `(1.0,0.9,0.0)` | `zone_A` |
| `st_B` | regular | 0.32 m | cyan `(0.0,0.9,0.9)`   | `zone_B` |
| `st_C` | heavy   | 0.52 m | purple `(0.7,0.0,0.9)` | `zone_C` |

- **Marker:** thin cylinder, radius ≈ 0.4 m, height 0.02 m, color = category. Floor level,
  in a reachable aisle in front of a rack. Defines the pickup XY. Visual + perception cue.
- **Pickable box:** a `RigidObjectCfg` box of the station's category at the marker XY, on a
  **low shelf / pedestal** at a reachable arm height (`pick_z ≈ 0.6–0.9 m` — within Panda's
  ~0.85 m vertical reach from the base, NOT the 1.5 m top shelf; verify against Panda
  workspace). The 18 top-shelf decoration boxes are untouched.
- **Placement constraint:** stations sit where the base can stop with the box inside the
  arm's reachable workspace. Pick a front-row island (y ≈ +8) with x spread, aisle clear of
  racks for the base footprint (Ridgeback ≈ 0.96 × 0.79 m).
- Station definitions live in a `STATION_SPECS` list (name, category, marker_xy,
  marker_color, box_size, box_pick_pose, zone_name) so the env + pickup manager read one
  source of truth.

## 6. Pickup State Machine (`env/pickup_manager.py`, approach A)

Robot-agnostic, per-env tensor buffers, called from `WarehouseRLEnv` post-physics-step.
This module owns NO Isaac asset knowledge beyond handles passed in — it is the unit under
test and the **seam for future robot swaps**.

**Per-env state**
- `phase`: int ∈ {`SEEKING`=0, `REACHING`=1, `CARRYING`=2, `DONE`=3}
- `target_cat`: int 0/1/2 — category sampled per episode (drives instruction + goal).
- `reach_timer`: int — frames elapsed in the scripted REACHING sequence.
- `carried_box`: int — index of the attached box (or -1).

**Transitions** (evaluated each control step)
```
SEEKING:
    if base_xy within R_marker(0.6 m) of stations[target_cat].marker
       and base_speed < 0.2 m/s:
        -> REACHING  (reach_timer = 0)

REACHING:   # scripted, base frozen, action ignored for ~K_reach steps (~15 @10Hz = 1.5s)
    t0..t_a : arm IK drives end-effector down to box grasp pose; fingers open
    t_a     : close fingers + KINEMATIC ATTACH box to gripper (box pose follows ee)
    t_a..K  : retract arm to carry pose (box held above base)
    if reach_timer >= K_reach: -> CARRYING

CARRYING:   # box pose written = gripper pose each step
    goal = stations[target_cat].zone_xyz
    if base_xy within R_zone(1.5 m) of matching zone: -> DONE

DONE:       # open fingers, release box at zone; episode success
```

**Goal exposure:** the manager writes `env.goal_pos` (existing buffer) each step =
marker xyz while SEEKING, zone xyz while CARRYING/DONE. Reward/obs read `env.goal_pos`
as today — no new plumbing.

**Mistake handling**
- Approaching a **wrong-category** marker (within R_marker of `stations[j]`, j≠target):
  per-step penalty; **no** state transition (only the correct marker triggers REACHING).
- Delivering to a **wrong zone** while CARRYING (within R_zone of zone k≠target):
  per-step penalty; no success, no termination (robot may still reach correct zone).

## 7. Scripted Arm Reach (inside REACHING)

- Use Isaac Lab `DifferentialIKController` to compute `panda_joint1..7` targets driving the
  end-effector to the box grasp pose (top-down approach). Box poses are known from
  `STATION_SPECS`, so a per-station precomputed joint trajectory is an acceptable fallback
  if IK proves fiddly.
- Gripper: command `panda_finger_joint.*` open→closed at the attach frame.
- **Kinematic attach:** from attach frame onward, each physics step set the box root pose =
  end-effector (gripper) world pose + grasp offset, and zero the box velocity. Same trick as
  the box-follow mechanic; reliable, no contact instability.
- During REACHING the **base is frozen**: zero base joint velocity targets, ignore policy
  action for those frames. Robot does not translate while the arm works.

## 8. Observations / Reward / Termination

**Obs** (`warehouse_env.py` ObservationsCfg)
- Add `carrying = ObsTerm(func=carrying_flag)` returning `env.pickup.carrying` as `(B,1)`.
- `goal` term unchanged in code (reads `env.goal_pos`, now phase-dependent).

**Rewards** (`warehouse_reward.py`, weights tunable in Pass)
| Term | Trigger | Sign/scale (initial) |
|---|---|---|
| `shaping` | dist(base, active goal) | −0.01 × dist |
| `pickup_success` | SEEKING→REACHING (correct grasp), once | +5 (one-shot) |
| `wrong_marker_pen` | near wrong marker, per step | −0.5 |
| `delivery_success` | DONE (correct box at correct zone) | +10 |
| `wrong_zone_pen` | near wrong zone while carrying, per step | −0.5 |
| `time_pen` | per step | −0.005 |
| `collision` | base contact > 5 N (existing sensor) | −5 |

**Terminations** (`warehouse_env.py` TerminationsCfg)
- `success`: `phase == DONE`.
- `time_out`: existing.
- `bounds`: existing.
- (Drop the old nav-only `reached_goal` termination — success now means delivered.)

## 9. Files Touched

| File | Change |
|---|---|
| `env/warehouse_scene.py` | robot → RidgebackFranka; add `STATION_SPECS`, markers, pickable boxes |
| `env/pickup_manager.py` | **new** — state machine, scripted arm/grasp, goal writing |
| `env/warehouse_env.py` | base `(2,)`→base-joint mapping; `carrying` obs; wire pickup manager into step/reset; reward/termination cfg |
| `env/warehouse_reward.py` | new reward funcs (pickup, wrong-marker, delivery gated on carrying, wrong-zone) |
| `configs/env_config.yaml` | new params (station coords, radii, K_reach, weights) |
| `tests/test_pickup_manager.py` | **new** — pure-python state-machine transition tests (no Isaac Sim) |
| `tests/test_env.py` | update obs contract assertions (+`carrying`) |
| `docs/environment.md` | document pickup task + new obs key |
| `CLAUDE.md` | update mission + obs contract + reward |

## 10. Testing Strategy

- **Pure unit (no Isaac Sim)** — like `test_layout_grid.py`: drive `PickupManager` with
  synthetic base positions/speeds through every transition; assert phase changes, goal
  switching, carrying flag, mistake penalties fire correctly. This is the primary safety net.
- **Smoke (Isaac Sim)** — `explore_scene.py`-style: spawn RidgebackFranka, confirm it loads,
  log base/arm joint + body names, eyeball one scripted reach+grasp on a station.
- **Contract** — `test_env.py`: assert obs dict has `carrying`, action still `(2,)`.

## 11. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| VRAM blowup (arm articulation, 8 GB) | med | headless, `num_envs=1`, lower physics iters; smoke test early |
| Blackwell SDP camera crash (open bug) | high | unchanged risk; camera on base. Pickup manager works headless w/o sensor cam for unit tests. Real fix tracked in `bugs_errors/2026-05-22_sdp-camera-crash-blackwell.md` |
| Arm IK fiddly / unreachable box pose | med | box at reachable `pick_z` (≤0.9 m); per-station precomputed joint trajectory fallback |
| Base frozen during REACHING feels abrupt | low | tune `K_reach`; acceptable for v1 |
| Sparse pickup reward → slow convergence | med | dense `shaping` to active goal + one-shot `pickup_success`; tune weights in a Pass |
| Holonomic base leaks strafe | low | hard-zero `prismatic_y` target every step |

## 12. Escalation Seam (future, NOT this spec)

`PickupManager` is robot-agnostic. A future "learned arm" or "physics grasp" upgrade
replaces only the REACHING implementation; stations, phases, goal-switch, reward, and obs
stay. Likewise a learned-grasp action upgrade would add action dims behind a flag without
touching navigation reward.
