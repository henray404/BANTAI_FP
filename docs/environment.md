# Warehouse Environment — Design Document

**Project**: Visual Goal-Conditioned World Model for Warehouse Pickup (pure DL — CLIP/YOLO removed 2026-06-08)
**Person**: P1 — Environment & Integration
**Stack**: Isaac Lab 5.1 · Python 3.11 · RTX 5050 8GB · CUDA 12.8
**Last updated**: 2026-06-08 (pickup redesign — see `docs/superpowers/specs/2026-06-08-pure-dl-pickup-redesign.md`)

> ⚠️ **2026-06-08 REDESIGN.** Task changed nav-only → pick→carry→place. CLIP + YOLO removed.
> Box category given via `goal_id` one-hot (no detection, no text). Arm now ACTIVE (DifferentialIKController).
> Sections below being migrated; spec is source of truth.

---

## 1. Mission

Robot spawns in the receiving area (north). A `goal_id` one-hot selects a category → it commands both the target box (by size) and the matching color zone. Robot navigates to the box, grasps it, carries it to the matching zone (south), and places it. Episode ends on delivery success or timeout (100s).

```
SPAWN (receiving, north)
        |
        v
read goal_id one-hot (A=orange / B=cyan / C=purple) -> target box + target zone
        |
        v
navigate through 9 rack islands to commanded box
        |
        v
grasp box (Franka arm, DifferentialIKController) -> holding=1, +5
        |
        v
carry to matching color zone, place -> +10, episode ends
        |
   timeout 100s -> truncated
```

**Task = pick→carry→place**, single box per episode. Arm ACTIVE.

**Category given directly via `goal_id`** (no CLIP text, no YOLO detection). Mapping:
| Zone | Color | Box category / size |
|---|---|---|
| zone_A | orange | fragile / 21cm |
| zone_B | cyan | regular / 32cm |
| zone_C | purple | heavy / 52cm |

---

## 2. Scene Layout

Room: **20 x 30 m** (x in [-10,+10], y in [-15,+15]).

```
NORTH WALL  y=+15
=========== RECEIVING (y +11..+14) ===================
  [forklift]   [pallet-asm]   [pallet-asm]
        ROBOT SPAWN AREA
------------------------------------------------------
  [I1]         [I2]         [I3]       y = +8
 [r] [r]      [r] [r]      [r] [r]
  [I4]         [I5]         [I6]       y = +1
 [r] [r]      [r] [r]      [r] [r]
  [I7]         [I8]         [I9]       y = -5
 [r] [r]      [r] [r]      [r] [r]
  x=-6         x=0          x=+6
        [pallet][pallet]   [cone][cone]
=========== SHIPPING (y -9..-14) =====================
  [ZONE_A]     [ZONE_B]     [ZONE_C]   y = -12
   orange       cyan         purple
  (-6,-12)     (0,-12)      (+6,-12)
    [sign]                    [sign]
SOUTH WALL  y=-15
```

### Room

| Param | Value |
|---|---|
| Size | 20 x 30 m |
| ROOM_HALF_X | 10.0 m |
| ROOM_HALF_Y | 15.0 m |
| Wall height | 6.0 m |
| Wall thickness | 0.3 m |
| env_spacing | 32.0 m (>30m + buffer) |

Walls: primitive concrete cuboids (building shell USD deferred — SubUSDs 1.2GB).

### Storage — 9 Island Blocks (3x3 grid)

| Param | Value |
|---|---|
| Island columns X | {-6, 0, +6} m |
| Island rows Y | {+8, +1, -5} m |
| Racks per island | 2 (side by side along X) |
| Total racks | 18 |
| ISLAND_RACK_DX | 1.5 m (tune after explore_rack.py — must > rack footprint_x) |
| Rack USD | Rack_L01_PR_NVD_01.usd (27MB; external MDL not copied -> pink/default material, non-fatal) |
| Rack scale | (0.01, 0.01, 0.01) — NVIDIA DT assets authored in cm, no auto-convert in Isaac Lab |
| Rack sim height | TBD — run explore_rack.py (was ~2.0m for old Rack_A) |
| RACK_SHELF_Z | 1.5 m — RETUNE after explore_rack.py (carried over from Rack_A) |

### Items (Graspable Boxes)

~18 boxes, one per rack on the **bottom shelf** (z≈0.72m surface), category cycling by rack index (fragile/regular/heavy). **Rigid bodies** (physics ON) — must be graspable. Bottom shelf only: Franka (mounted ~0.5m up on Ridgeback, reach ~0.85m) covers z≈0–1.35m, so bottom (0.72m) is reachable, mid (1.32m) borderline, top (1.93m) out of reach. Reachability is a hard constraint.

| Category | USD | Size | Zone |
|---|---|---|---|
| fragile | CubeBox_A03_21cm_PR_NVD_01.usd | 0.21 m | zone_A (orange) |
| regular | CubeBox_A05_32cm_PR_NVD_01.usd | 0.32 m | zone_B (cyan) |
| heavy | CubeBox_A07_52cm_PR_NVD_01.usd | 0.52 m | zone_C (purple) |

Size encodes category. Category given directly via `goal_id` one-hot — NOT detected. Target box is grasped and delivered.

### Delivery Zones (Goals)

Flat 3x3m colored slabs at south. These are the robot's goal targets.

| Zone | Color (RGB) | Position | Category |
|---|---|---|---|
| zone_A | orange (1.0, 0.9, 0.0) | (-6, -12, 0.01) | fragile |
| zone_B | cyan (0.0, 0.9, 0.9) | (0, -12, 0.01) | regular |
| zone_C | purple (0.7, 0.0, 0.9) | (+6, -12, 0.01) | heavy |

### Props (Realism + Obstacle)

All static USD assets, scale (0.01, 0.01, 0.01).

| Prop | Count | Placement |
|---|---|---|
| Forklift_A01 | 1 | (-5, +12) — receiving, static obstacle |
| Pallet_Asm_A01 (66x66x46cm boxes on pallet) | 1 | (+3, +12) — receiving staging |
| Pallet_Asm_A04 (120x122x75cm boxes on pallet) | 1 | (+5, +12) — receiving staging |
| EconomyPlasticPallet | 2 | (-3,-7) and (+4,-7) — aisle clutter |
| HeavyDutyTrafficCone 71cm | 4 | Aisle corners near islands |
| WarningSign_A01 | 2 | (-6,-9.5) and (+6,-9.5) — near zones |

---

## 3. Robot

> 🔧 **Diperbarui 2026-06-08.** Robot diganti dari Jetbot/Carter ke Ridgeback-Franka mobile
> manipulator (2026-06-01). **Tidak ada wheel kinematics** — base holonomik dipaksa diff-drive
> lewat dummy joint velocity.

| Param | Value |
|---|---|
| Model | Ridgeback-Franka (Clearpath holonomik + Franka Panda 7-DOF + gripper; Isaac 5.1 Nucleus USD) |
| Type | Holonomic base **dipaksa diff-drive** (kontrak action tidak berubah) |
| Joints | 3 dummy base (`prismatic_x/y` + `revolute_z`) + 7 arm (`panda_joint1..7`) + 2 finger = 12 |
| Spawn | Receiving-north: x[-8,+8], y[+11,+14], yaw random [-π,π] |
| Arm | **ACTIVE** — driven via DifferentialIKController (top-down EE). Pickup task approved 2026-06-08. |

### Action Space

```python
action = [base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper]   # shape (6,), values in [-1, 1]
```

Base mapping di `warehouse_env.py::_base_cmd` (body-frame yaw projection, BUKAN wheel kinematics):
```python
# base prismatic = world-frame → proyeksi yaw supaya "maju" ikut heading
vx = base_lin * cos(yaw)
vy = base_lin * sin(yaw)     # base dipaksa diff-drive; strafe murni tidak dipakai
wz = base_ang
# drive 3 dummy base joints: [dummy_base_prismatic_x, _y, dummy_base_revolute_z] = [vx, vy, wz]
```

Arm mapping: `(ee_dx, ee_dy, ee_dz)` = EE position delta in base frame → DifferentialIKController → `panda_joint1..7`. EE orientation fixed top-down. `gripper`: >0 open, ≤0 close (`panda_finger_joint1/2`).

| Param | Value | Source |
|---|---|---|
| MAX_LIN_SPEED | 1.5 m/s | warehouse_env.py |
| MAX_ANG_SPEED | 1.5 rad/s | warehouse_env.py |

> ⚠️ Tanpa wheel kinematics (`WHEEL_BASE`/`WHEEL_RADIUS` tidak relevan untuk Ridgeback).
> Catatan: base bisa keluar bounds <1 dtk @1.5 m/s — perlu action smoothing/effort tuning sebelum training serius.
> Arm IK: copy reference envs `Isaac-Reach-Franka-v0` / `Isaac-Lift-Cube-Franka-v0`.

---

## 4. Camera

| Param | Value |
|---|---|
| Type | **TiledCameraCfg** (SDP crash di RTX 5050 Blackwell RESOLVED via driver 580.88, 2026-06-03 — see bugs_errors/) |
| Resolution | 64 x 64 pixels |
| Channels | RGB, float [0,1] |
| Update period | 0.1s (10Hz) |
| FOV | ~60° (focal_length 18mm) |
| Offset from robot | (0.05, 0.0, 0.10) m, forward-facing |

64x64 is DreamerV3 standard across all 150+ benchmarks.

---

## 5. Observation Space — Interface Contract

```python
obs = {
    # --- navigation ---
    "pixels":   Tensor(batch, 3, 64, 64),   # camera RGB, float [0,1]
    "position": Tensor(batch, 3),            # robot base xyz, env-local
    "heading":  Tensor(batch, 2),            # [cos(yaw), sin(yaw)]
    "goal":     Tensor(batch, 3),            # delivery zone xyz (anneals to zeros)
    "goal_id":  Tensor(batch, 3),            # one-hot [orange,cyan,purple] — selects box + zone (replaced goal_emb 2026-06-08)
    # --- manipulation ---
    "ee_pos":   Tensor(batch, 3),            # end-effector xyz, base frame
    "gripper":  Tensor(batch, 1),            # finger opening 0..1
    "holding":  Tensor(batch, 1),            # 1.0 if target box grasped
    "box_pos":  Tensor(batch, 3),            # target box xyz, env-local — UNANNEALED
}
```

**DO NOT CHANGE this interface without team discussion.**

### Why no velocity observation?

Council review (3 independent voices) + TEEP RMFS (2025): DreamerV3's RSSM is recurrent — it reconstructs motion implicitly from pixel sequences. Explicit velocity is redundant. (Proprioception `ee_pos`/`gripper`/`holding` is kept because arm state is not inferable from base-mounted pixels.)

### goal xyz — Curriculum plan

`goal` (delivery zone xyz) anneals to zeros so the policy relies on `goal_id` + pixels for delivery. `box_pos` stays UNANNEALED (grasp needs precision).

```
Phase 1: goal = zone xyz           <- easy baseline, debug env
Phase 2: goal = zone xyz * alpha   <- anneal alpha 1.0 -> 0.0
Phase 3: goal = zeros              <- robot relies on goal_id + pixels
```

---

## 6. Reward Function

**Staged pick-place** (switches on `holding` flag) — see spec §4.

```python
# Phase A — approach + grasp (NOT holding)
- 0.01 * dist(ee_pos, box_pos)        # dense: hand -> target box
+ 5.0  * grasp_success                # box gripped + lifted off shelf
# Phase B — carry + place (holding)
- 0.01 * dist(box_pos, goal_zone)     # dense: box -> correct color zone
+ 10.0 * delivery_success             # correct box in correct color zone, released
# Always-on
- 0.005 * time_penalty                # efficiency
- 5.0   * collision                   # chassis contact > 5N
- 2.0   * drop_penalty                # box dropped mid-carry outside zone
```

- `grasp_success` (+5 intermediate) bootstraps learning — policy gets signal before full chain. Pure-sparse rejected (too hard).
- `delivery_success` requires category→color match (`goal_id`). Wrong zone = no reward.

**Rule**: land each reward change alone. Never change two things at once — can't diagnose which broke training.

---

## 7. Termination Conditions

| Condition | Trigger |
|---|---|
| `time_out` | 100s = 1000 steps @ 10Hz |
| `delivery_success` | correct box (per `goal_id`) inside matching color zone, released |
| `out_of_bounds` | robot outside x +-9.5m or y +-14.5m |

---

## 8. Simulation Parameters

| Param | Value | Rationale |
|---|---|---|
| Physics dt | 0.005s (200Hz) | Isaac Lab standard |
| Decimation | 20 | 200/20 = 10Hz policy (DreamerNav standard) |
| Episode | 100s x 10Hz = 1000 steps | nav + grasp + carry + place horizon |
| num_envs | 2 (default) | VRAM-safe on 8GB for 30m room + props |
| env_spacing | 32.0m | > 30m largest room dim + 2m buffer |

10Hz control grounded in DreamerNav (2025): "10 Hz" policy with "1 m/s" robot speed, successfully navigating avg 18.56m per episode.

### VRAM Estimate (RTX 5050 8GB, num_envs=2)

| Component | Estimate |
|---|---|
| PhysX + scene (2 envs) | ~3.0 GB |
| TiledCamera 64x64 (2 envs) | ~0.2 GB |
| DreamerV3 model (P2) | ~1.5-2.0 GB |
| Replay buffer | ~0.3 GB |
| **Total** | **~5.0-5.5 GB** |

If stable at num_envs=2 → try 4. If OOM → keep at 2.

---

## 9. Physics

Ground friction: static=0.8, dynamic=0.6, restitution=0.0 (explicit, not PhysX defaults).

USD scale (0.01, 0.01, 0.01) applied to all NVIDIA DT assets — Isaac Lab's UsdFileCfg has no automatic metersPerUnit conversion. All NVIDIA warehouse assets authored in centimeters. Without scale, a 2m rack spawns at 200m.

---

## 10. Code Structure

```
env/
  layout_grid.py    # Pure-python grid math: island_rack_positions(), item_specs()
                    # No Isaac import -> pytest runs without GPU. TDD-tested.
  warehouse_scene.py  # Scene: inherits layout_grid, spawns assets via __post_init__
  warehouse_env.py    # MDP: obs/action/reward/termination + Gymnasium wrapper
  warehouse_reward.py # reward functions: delivery_success, distance_to_goal, out_of_bounds

tests/
  test_layout_grid.py  # Pure unit tests (run: pytest tests/test_layout_grid.py)
  test_env.py          # Isaac Sim integration tests (needs GPU + conda activate isaaclab)
```

---

## 11. How to Run

```bash
conda activate isaaclab

# Visual inspection — tune RACK_SHELF_Z and ISLAND_RACK_DX
python asset_sandbox/scripts/explore_rack.py   # prints rack X/Y/Z footprint
python asset_sandbox/scripts/explore_scene.py  # opens scene viewport

# Interface contract test
python tests/test_env.py --num_envs 1

# Headless training run
python scripts/run_env.py --num_envs 2 --headless --steps 99999

# Pure layout math tests (no GPU needed)
pytest tests/test_layout_grid.py -v
```

---

## 12. Pending Work (ordered by pass)

### Pass 2 — Verify scene (YOU need to run these)
- [ ] `explore_rack.py` -> print footprint_x -> tune `ISLAND_RACK_DX` if racks overlap
- [ ] `explore_scene.py` -> visual check: 9 islands, boxes on shelves, props correct, no floating
- [ ] `test_env.py --num_envs 1` -> all PASS
- [ ] `run_env.py --num_envs 2 --headless --steps 200` -> no OOM

### Pass 3 — Collision reward (after Pass 2 verifies collision geometry fires)
- [ ] Add ContactSensor on robot
- [ ] Collision penalty ~-50 (not -0.1)
- [ ] Validate reward fires (non-zero near walls/racks)

### Pass 4 — Reward rebalance (land alone, behind flag)
- [ ] Goal reward scale up (+10..+100)
- [ ] Heading-alignment term: cos(theta_goal)
- [ ] Potential-based distance shaping
- [ ] Time penalty scale up (-0.1..-0.5)
- [ ] 1k-step sign-sanity run; raw-dist fallback flag

### Pass 5 — Curriculum
- [ ] DreamerNav 6-level goal-distance curriculum ([2-5]m -> [2-inf]m)
- [ ] Success radius curriculum: 1.4m -> 1.0m
- [ ] goal xyz anneal: alpha 1.0 -> 0.0 (box_pos stays unannealed)

### Pickup migration (2026-06-08 redesign — NEW work)
- [ ] Boxes static → rigid bodies, placed within Franka reach (~0.85m)
- [ ] Action space (2,) → (6,): add `ee_dx/dy/dz, gripper`
- [ ] Wire DifferentialIKController (copy `Isaac-Lift-Cube-Franka-v0`)
- [ ] Obs: add `ee_pos, gripper, holding, box_pos`; replace `goal_emb` → `goal_id`
- [ ] Reward: staged grasp/place + `grasp_success`/`drop_penalty`
- [ ] Grasp-success detection (finger contact + box lift threshold)
- [ ] VRAM re-measure (~18 boxes + arm IK)

### Deferred
- Building shell (Warehouse_A modular) — SubUSDs 1.2GB, too heavy
- 6-DOF EE orientation control (v1 = fixed top-down)
- cuRobo motion planning (v1 = DifferentialIKController only)
- Domain randomization (friction, lighting)

---

## 13. Research Grounding

| Decision | Source |
|---|---|
| Navigate-to-zone mission | DreamerNav (2025), TEEP RMFS (2025) |
| 64x64 RGB | DreamerV3 Hafner et al. 2023 (all 150+ benchmarks) |
| 10Hz control, 1 m/s | DreamerNav (2025) Table 7 |
| 1000 steps / 100s episode | nav + grasp + carry + place needs longer horizon than nav-only |
| No velocity obs | Council 3-voice + TEEP RMFS (trains without it) |
| goal xyz anneal curriculum | Council Decision F |
| Collision >> goal reward | DreamerNav (-50 vs +100), TEEP (-3 vs +2) |
| Heading cos(theta) reward | DreamerNav Table 7 |
| Island-block layout | TEEP reference image + user preference |
| Items rigid (graspable) | Pickup task (2026-06-08); boxes must be grasped — physics ON |
| scale=(0.01,...) on USDs | Measured: Isaac Lab UsdFileCfg has no auto metersPerUnit conversion |

Full paper list: `docs/referensi.md`

---

## 14. Known Bugs (all fixed)

| Bug | Status |
|---|---|
| CameraCfg crash on RTX 5050 Blackwell | FIXED: TiledCameraCfg |
| USD box items fall off shelves on reset | FIXED: static AssetBaseCfg |
| Robot speed effectively 0.32 m/s (wrong ACTION_SCALE) | FIXED: explicit MAX_LIN_SPEED=1.0 |
| Rack/box USD 100x too large (cm units, no scale) | FIXED: scale=(0.01,0.01,0.01) |
| Double AppLauncher crash | FIXED: AppLauncher only in entry scripts |

See `bugs_errors/` for full details.
