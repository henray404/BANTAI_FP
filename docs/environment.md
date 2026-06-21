# Warehouse Environment — Design Document

**Project**: Text-Conditioned World Model for Visual Category-Aware Warehouse Robot
**Person**: P1 — Environment & Integration
**Stack**: Isaac Lab 5.1 · Python 3.11 · RTX 5050 8GB · CUDA 12.8
**Last updated**: 2026-05-30 (Layout v2)

---

## 1. Mission

Robot spawns in the receiving area (north). A random delivery zone is assigned as the goal. Robot navigates through the rack-island storage field to reach the correct colored zone (south). Episode ends on success (within 0.5m of zone) or timeout (60s).

```
SPAWN (receiving, north)
        |
        v
read goal zone (A=orange / B=cyan / C=purple)
        |
        v
navigate through 9 rack islands, avoid obstacles
        |
        v
reach zone within 0.5m radius -> +1, episode ends
        |
   timeout 60s -> truncated
```

**Phase 1 = navigation only.** No item pickup (no manipulator). Items are visual landmarks for CLIP/YOLO.

**Text instruction per zone** (for Person 4 CLIP encoding):
| Zone | Color | Instruction |
|---|---|---|
| zone_A | orange | "deliver small box to orange zone" |
| zone_B | cyan | "deliver medium box to cyan zone" |
| zone_C | purple | "deliver large box to purple zone" |

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

### Items (Visual Landmarks)

18 boxes, one per rack, category cycling by rack index (fragile/regular/heavy). **Static props** — no physics, stay on shelves across episode resets.

| Category | USD | Size | Zone |
|---|---|---|---|
| fragile | CubeBox_A03_21cm_PR_NVD_01.usd | 0.21 m | zone_A (orange) |
| regular | CubeBox_A05_32cm_PR_NVD_01.usd | 0.32 m | zone_B (cyan) |
| heavy | CubeBox_A07_52cm_PR_NVD_01.usd | 0.52 m | zone_C (purple) |

Size encodes category for CLIP/YOLO. Items are NOT picked up — navigation task only.

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

| Param | Value |
|---|---|
| Model | Jetbot (NVIDIA Nucleus USD, placeholder for custom AMR) |
| Type | Wheeled differential-drive |
| Spawn | Receiving-north: x[-8,+8], y[+11,+14], yaw random [-π,π] |

### Action Space

```python
action = [linear_vel, angular_vel]   # shape (2,), values in [-1, 1]
```

Conversion in `WarehouseGymEnv._diff_drive()`:
```python
lin_mps = action[:,0] * 1.0    # max 1.0 m/s
ang_rps = action[:,1] * 2.0    # max 2.0 rad/s
v_left  = (lin_mps - 0.5 * 0.118 * ang_rps) / 0.032   # rad/s -> left wheel
v_right = (lin_mps + 0.5 * 0.118 * ang_rps) / 0.032   # rad/s -> right wheel
```

| Param | Value | Source |
|---|---|---|
| MAX_LIN_SPEED | 1.0 m/s | DreamerNav (2025) standard |
| MAX_ANG_SPEED | 2.0 rad/s | — |
| WHEEL_BASE | 0.118 m | Jetbot spec |
| WHEEL_RADIUS | 0.032 m | Jetbot spec |

---

## 4. Camera

| Param | Value |
|---|---|
| Type | **TiledCameraCfg** (CameraCfg crashes on RTX 5050 Blackwell — see bugs_errors/) |
| Resolution | 64 x 64 pixels |
| Channels | RGB, float [0,1] |
| Update period | 0.1s (10Hz) |
| FOV | ~48.7° |
| Offset from robot | (0.05, 0.0, 0.10) m, forward-facing |

64x64 is DreamerV3 standard across all 150+ benchmarks.

---

## 5. Observation Space — Interface Contract

```python
obs = {
    "pixels":   Tensor(batch, 3, 64, 64),   # camera RGB, float [0,1]
    "position": Tensor(batch, 3),            # robot xyz, env-local
    "goal":     Tensor(batch, 3),            # target zone xyz (curriculum: will anneal to zeros)
    "goal_emb": Tensor(batch, 512),          # CLIP embedding — zeros until P4 wires it
}
```

**DO NOT CHANGE this interface without team discussion.**

### Why no velocity observation?

Council review (3 independent voices) + TEEP RMFS (2025) = unanimous decision:
DreamerV3's RSSM is recurrent — it reconstructs motion implicitly from pixel sequences. Adding an explicit velocity vector is redundant and would break the P2/P4 interface contract. TEEP trained a warehouse world model successfully without velocity.

### goal xyz — Curriculum plan (Pass 5)

The `goal` xyz leaks the exact target location, which short-circuits CLIP/language learning. Plan:

```
Phase 1 (now):   goal = zone xyz           <- easy baseline, debug env
Phase 2 (Pass 5): goal = zone xyz * alpha  <- anneal alpha from 1.0 to 0.0
Phase 3 (final): goal = zeros              <- robot relies on goal_emb (CLIP) only
```

Coordinate with P4 when wiring CLIP.

---

## 6. Reward Function

**Current (Pass 1 — stable baseline, do not change until Pass 3)**

```python
reward = +1.0 * delivery_success    # sparse: within 0.5m of goal zone
       - 0.05 * dist(robot, goal)   # dense shaping
       - 0.001 * time_penalty       # per step
```

### Upcoming Rebalance (Pass 3 + Pass 4)

Grounded in DreamerNav (2025) Table 7 and TEEP RMFS (2025):

| Term | DreamerNav ref | Current | Target |
|---|---|---|---|
| Goal reward | +100 | +1 | +10..+100 |
| Collision | -50 | 0 | **-50 (current -0.1 plan is negligible, use -50)** |
| Heading align | cos(theta_goal) | missing | add cos(theta_goal) |
| Time penalty | -0.5/step | -0.001 | -0.1..-0.5 |
| Distance shaping | +5 subgoal | -0.05*dist | potential-based delta |

**Rule**: land each reward change alone. Never change two things at once — can't diagnose which broke training.

---

## 7. Termination Conditions

| Condition | Trigger |
|---|---|
| `time_out` | 60s = 600 steps @ 10Hz |
| `reached_goal` | robot within 0.5m of goal zone |
| `out_of_bounds` | robot outside x +-9.5m or y +-14.5m |

---

## 8. Simulation Parameters

| Param | Value | Rationale |
|---|---|---|
| Physics dt | 0.005s (200Hz) | Isaac Lab standard |
| Decimation | 20 | 200/20 = 10Hz policy (DreamerNav standard) |
| Episode | 60s x 10Hz = 600 steps | ~25m traverse @1m/s + island navigation |
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
- [ ] goal xyz anneal: alpha 1.0 -> 0.0 (coordinate with P4 CLIP wiring)

### Deferred
- Building shell (Warehouse_A modular) — SubUSDs 1.2GB, too heavy
- Custom AMR swap (Jetbot is placeholder)
- Domain randomization (friction, lighting)
- Pickup mechanic (Phase 2, needs manipulator)

---

## 13. Research Grounding

| Decision | Source |
|---|---|
| Navigate-to-zone mission | DreamerNav (2025), TEEP RMFS (2025) |
| 64x64 RGB | DreamerV3 Hafner et al. 2023 (all 150+ benchmarks) |
| 10Hz control, 1 m/s | DreamerNav (2025) Table 7 |
| 600 steps / 60s episode | DreamerNav max 400 steps @10Hz = 40s, ours slightly longer for 30m room |
| No velocity obs | Council 3-voice + TEEP RMFS (trains without it) |
| goal xyz anneal curriculum | Council Decision F |
| Collision >> goal reward | DreamerNav (-50 vs +100), TEEP (-3 vs +2) |
| Heading cos(theta) reward | DreamerNav Table 7 |
| Island-block layout | TEEP reference image + user preference |
| Items static (not rigid) | Navigation task only; no manipulator; static = no reset-fall bug |
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
