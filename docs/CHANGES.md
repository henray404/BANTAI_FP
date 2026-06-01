# Environment Improvement Changes — 2026-06-01

Deep-audit implementation based on physics, layout, robot, and reward analysis.
All changes by Person 1 (Henry). Coordinate with P2 (DreamerV3) on new obs key `heading`.

---

## Summary Table

| ID | Change | File(s) | Priority |
|---|---|---|---|
| P0.1 | Add ContactSensorCfg | warehouse_scene.py | P0 |
| P0.2 | Fix shaping reward sign convention | warehouse_reward.py + warehouse_env.py | P0 |
| P0.3 | Replace Jetbot → Carter v2.4 | warehouse_scene.py + warehouse_env.py | P0 |
| P1.1 | Success threshold 0.5m → 1.5m | warehouse_reward.py | P1 |
| P1.2 | Add heading observation | warehouse_env.py | P1 |
| P1.3 | Rebalance reward weights | warehouse_env.py | P1 |
| P1.4 | Camera FOV: focal_length 24→18mm | warehouse_scene.py | P1 |
| P1.5 | Remove dead mass param from `_item_cfg` | warehouse_scene.py | P1 |
| P2.3 | Wheel friction material on Carter | warehouse_scene.py | P2 |
| P2.5 | Lower max_depenetration_velocity 5.0→1.0 | warehouse_scene.py | P2 |
| DOC | Update CLAUDE.md interface contract | CLAUDE.md | - |
| DOC | Sync env_config.yaml | configs/env_config.yaml | - |

---

## Detailed Changes

### P0.1 — Add ContactSensorCfg to Scene

**File:** `env/warehouse_scene.py`

**Problem:** No contact sensor — Phase 3 collision penalty impossible.

**Change:**
- Added `ContactSensorCfg` import
- Added `contact_sensor` field to `WarehouseSceneCfg`, monitoring `Robot/chassis_link`
- Filters against `Rack_.*` and `wall_.*` prims

```python
contact_sensor: ContactSensorCfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/chassis_link",
    update_period=0.0,
    history_length=3,
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Rack_.*", "{ENV_REGEX_NS}/wall_.*"],
)
```

> **Note:** `chassis_link` is the expected Carter v2.4 base prim name. If it fails at runtime, run `explore_scene.py` and check the USD hierarchy to find the correct prim name.

---

### P0.2 — Fix Shaping Reward Sign Convention

**Files:** `env/warehouse_reward.py`, `env/warehouse_env.py`

**Problem:** `distance_to_goal()` returned *negative* distance with *positive* weight — correct math but buried sign was a maintenance hazard.

**Before:**
```python
def distance_to_goal(...): return -torch.norm(...)   # negative in func
shaping = RewTerm(func=distance_to_goal, weight=0.05)   # positive weight
```

**After:**
```python
def distance_to_goal(...): return torch.norm(...)    # positive in func
shaping = RewTerm(func=distance_to_goal, weight=-0.01)  # sign explicit in weight
```

---

### P0.3 — Replace Jetbot → NVIDIA Carter v2.4

**Files:** `env/warehouse_scene.py`, `env/warehouse_env.py`, `configs/env_config.yaml`

**Problem:** Jetbot is an 11.3cm × 11.3cm toy robot (~500g).
- Occupies 3% of 3.7m aisle — barely visible in 64×64 camera
- Camera at 10cm height sees mostly floor
- Real max speed 0.32 m/s vs configured 1.0 m/s (10× overdriven)
- `effort_limit=100 N·m` on 500g robot — explosive collision forces

**Replacement: Carter v2.4**

| Spec | Jetbot (was) | Carter v2.4 (now) |
|---|---|---|
| Body width | 113mm | 580mm |
| Body length | 113mm | 660mm |
| Weight | ~500g | 22 kg |
| Wheel base | 0.118m | **0.570m** |
| Wheel radius | 0.032m | **0.097m** |
| Max speed | 0.32 m/s | 2.0 m/s (configured 1.5) |
| Camera height | 10cm | **50cm** |
| USD path | `Robots/NVIDIA/Jetbot/jetbot.usd` | `Robots/NVIDIA/Carter/carter_v2.4.usd` |

**Physics improvements in CARTER_CFG:**
- `max_depenetration_velocity=1.0` (was 5.0)
- `enable_gyroscopic_forces=True`
- `RigidBodyMaterialCfg`: static_friction=0.9, dynamic_friction=0.7, restitution=0.0, friction_combine_mode="multiply"
- `effort_limit_sim=20.0 N·m` (was 100.0)
- `velocity_limit_sim=25.0 rad/s` → max 2.4 m/s (was 50.0)

**Constants updated in warehouse_env.py:**
```python
WHEEL_BASE    = 0.570   # was 0.118
WHEEL_RADIUS  = 0.097   # was 0.032
MAX_LIN_SPEED = 1.5     # m/s (was 1.0)
MAX_ANG_SPEED = 1.5     # rad/s (was 2.0)
```

**`_validate()` updated:** `n_joints >= 2` (was `== 2`) because Carter may expose passive caster joints.

---

### P1.1 — Success Threshold 0.5m → 1.5m

**File:** `env/warehouse_reward.py`

**Problem:** 0.5m on a 3×3m zone = pinpoint target. Too hard for early training.

**Change:** `threshold=1.5m` in both `delivery_success()` and `reached_goal()`.
- 1.5m = zone edge from center (3×3m zone → ±1.5m)
- Robot succeeds when it enters the zone footprint

---

### P1.2 — Add Heading Observation

**File:** `env/warehouse_env.py`

**Problem:** Robot spawns random yaw (-π to +π). Without heading, policy cannot distinguish "facing north" from "facing south" at same (x,y) — only pixels differentiate. Severely slows early learning.

**Change:** New `robot_heading()` → `[cos(yaw), sin(yaw)]` (unit-circle, no ±π discontinuity).

```python
obs["heading"] = Tensor(batch, 2)   # [cos(yaw), sin(yaw)]
```

**Interface contract updated in CLAUDE.md. P2 (DreamerV3) must handle this new obs key.**

---

### P1.3 — Rebalance Reward Weights

**File:** `env/warehouse_env.py`

**Problem:** Over 600 steps from spawn, shaping total ≈ -750 vs success = +1 once. Ratio 750:1 — success invisible to value estimator.

| Term | Before | After | Reason |
|---|---|---|---|
| `success` | weight=1.0 | weight=**10.0** | Raise to balance shaping |
| `shaping` | weight=0.05 (implicit -) | weight=**-0.01** | Reduce + fix sign |
| `time_pen` | weight=-0.001 | weight=**-0.005** | More efficiency pressure |
| `collision` | *(missing)* | weight=**5.0** | New (needs ContactSensor P0.1) |

Collision penalty function (in warehouse_reward.py):
```python
def collision_penalty(...):
    # returns 0 or -1; weight=5.0 → effective penalty=-5 per collision step
```

---

### P1.4 — Wider Camera FOV

**File:** `env/warehouse_scene.py`

| Parameter | Before | After |
|---|---|---|
| focal_length | 24.0mm | **18.0mm** |
| HFOV | 46.8° | **~60°** |

At 50cm height and 60° HFOV, robot sees full aisle width at 3m distance and rack-face boxes at 2–4m range.

---

### P1.5 — Remove Dead Mass Param from `_item_cfg`

**File:** `env/warehouse_scene.py`

**Problem:** `_item_cfg(name, size, mass, pos)` had `mass` parameter that `AssetBaseCfg` silently ignored — dead code.

**Change:**
- `_item_cfg` signature: removed `mass` param
- `__post_init__` loop: unpacks `_mass` (unused marker) instead of passing it
- `BOX_MASSES` constant **kept** as reference for Phase 3 (boxes → `RigidObjectCfg`)
- `layout_grid.py` and tests **unchanged** — still returns mass for future use

---

### P2.3 + P2.5 — Robot Physics Material + Depenetration Velocity

Applied inside `CARTER_CFG` (part of P0.3):
- `RigidBodyMaterialCfg` with rubber-on-concrete friction values and `friction_combine_mode="multiply"`
- `max_depenetration_velocity=1.0` — prevents ghost-through artifacts on hard collisions

---

## Team Coordination

### P2 (DreamerV3) — Action Required

New obs key `heading: Tensor(batch, 2)` in observation dict. Handle in RSSM input. If not ready, gate with zero-weight projection.

### P3 (YOLOv8)

Camera height 10cm → 50cm; HFOV 47° → 60°. Box USD assets unchanged. Detection confidence calibration at close range may shift.

### P4 (CLIP)

No change. `goal_emb` interface unchanged.

---

## Pending (Phase 2+)

| ID | Change | Trigger |
|---|---|---|
| P2.1 | Add velocity obs `[lin_vel, ang_vel]` | P2 sign-off |
| P2.2 | Add depth channel (3→4 channels) | P2 sign-off |
| P2.4 | Randomize box positions ±0.1m per reset | Phase 2 start |
| Verify | Confirm `chassis_link` prim name in Carter USD at first run | First run |
| Verify | Confirm `.*wheel.*` regex matches Carter joints at first run | First run |
| Verify | If boxes fall to floor (not shelf): recalibrate `RACK_SHELF_Z` via `explore_rack.py` | First run |

---

## 2026-06-01 (Follow-up) — Box Gravity Fix

**Problem:** Boxes floating (terbang) + clipping through racks (tembus) — `AssetBaseCfg` = static collider, no gravity.

**Fix:** `_item_cfg` converted to `RigidObjectCfg`.

| Property | Value |
|---|---|
| gravity | enabled (`disable_gravity=False`) |
| masses | fragile=2.0 kg, regular=6.0 kg, heavy=12.0 kg |
| max_depenetration_velocity | 1.0 m/s |
| contact_offset | 0.005 m |
| init z | +0.05m above computed shelf — falls naturally, avoids start-inside-geometry explosion |
| episode reset | automatic via `scene.reset()` — boxes return to init_state each episode |

If boxes still fall to floor instead of settling on shelf: shelf collision geometry missing or `RACK_SHELF_Z` wrong — recalibrate via `explore_rack.py`.
