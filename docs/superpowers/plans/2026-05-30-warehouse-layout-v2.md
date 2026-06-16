# Warehouse Layout v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the warehouse scene as a realistic 20x30m island-block warehouse (9 rack islands, 18 boxes, receiving/shipping props) while keeping the navigate-to-colored-zone mission and obs contract unchanged.

**Architecture:** Extract layout math into a pure-python module (`env/layout_grid.py`, unit-testable without Isaac). Scene config consumes it and spawns racks/items/props via `__post_init__` setattr. Env/reward params updated for the rectangular room. Existing explore scripts + test_env are the integration checks (run by user — Isaac Sim needs GPU).

**Tech Stack:** Isaac Lab 5.1, Python 3.11, pytest (pure unit tests), NVIDIA DT USD assets (cm-authored, scale 0.01).

**Spec:** `docs/superpowers/specs/2026-05-30-warehouse-layout-v2-design.md`

**Note on testing:** Scene/sim tasks can't run in a headless unit test (need Isaac Sim app + GPU). Only `env/layout_grid.py` is true-TDD. Scene/env/reward tasks are verified by USER running `explore_scene.py` / `test_env.py` (windowed). Each such task lists the exact command + expected output.

---

## Task 1: Copy NVIDIA props into project assets/

**Files:**
- Create: `assets/Equipment/Forklifts/Forklift_A/Forklift_A01_PR_V_NVD_01.usd` (+ its material/texture deps)
- Create: `assets/Shipping/Cardboard_Boxes_on_Pallet/Pallet_Asm_A/Pallet_Asm_A01_66x66x46cm_PR_V_NVD_01.usd` (+ A04)
- Create: `assets/Shipping/Pallets/Plastic/Economy_A/EconomyPlasticPallet_A01_PR_NVD_01.usd`
- Create: `assets/Safety/Cones/Heavy-Duty_Traffic/HeavyDutyTrafficCone_A02_71cm_PR_V_NVD_01.usd`
- Create: `assets/Safety/Floor_Signs/Warning_A/WarningSign_A01_PR_NVD_01.usd`

Source root: `C:\Users\Henry\Downloads\Warehouse_NVD@10013\Assets\DigitalTwin\Assets\Warehouse\`

- [ ] **Step 1: Copy each asset folder (with sibling materials/ + textures/)**

Run (PowerShell), copying the *containing folder* so self-contained material/texture siblings come along:

```powershell
$src = "C:\Users\Henry\Downloads\Warehouse_NVD@10013\Assets\DigitalTwin\Assets\Warehouse"
$dst = "C:\Users\Henry\Documents\KULIAH\sem_4\BANTAI_FP\assets"
# Rack: copy only L01 (folder has 18 variants ~700MB; we use one)
New-Item -ItemType Directory -Force "$dst\Shelving\Racks\Rack_L" | Out-Null
Copy-Item "$src\Shelving\Racks\Rack_L\Rack_L01_PR_NVD_01.usd" "$dst\Shelving\Racks\Rack_L\" -Force
Copy-Item "$src\Equipment\Forklifts\Forklift_A" "$dst\Equipment\Forklifts\Forklift_A" -Recurse -Force
Copy-Item "$src\Shipping\Cardboard_Boxes_on_Pallet\Pallet_Asm_A" "$dst\Shipping\Cardboard_Boxes_on_Pallet\Pallet_Asm_A" -Recurse -Force
Copy-Item "$src\Shipping\Pallets\Plastic\Economy_A" "$dst\Shipping\Pallets\Plastic\Economy_A" -Recurse -Force
Copy-Item "$src\Safety\Cones\Heavy-Duty_Traffic" "$dst\Safety\Cones\Heavy-Duty_Traffic" -Recurse -Force
Copy-Item "$src\Safety\Floor_Signs\Warning_A" "$dst\Safety\Floor_Signs\Warning_A" -Recurse -Force
```

- [ ] **Step 2: Verify files exist**

Run:
```powershell
Get-ChildItem -Recurse "C:\Users\Henry\Documents\KULIAH\sem_4\BANTAI_FP\assets" -Filter *.usd | Where-Object { $_.Name -match "Forklift_A01|Pallet_Asm_A0(1|4)|EconomyPlasticPallet_A01|HeavyDutyTrafficCone_A02|WarningSign_A01" } | Select-Object Name
```
Expected: 6 .usd names listed (Forklift_A01, Pallet_Asm_A01, Pallet_Asm_A04, EconomyPlasticPallet_A01, HeavyDutyTrafficCone_A02, WarningSign_A01).

- [ ] **Step 3: Commit**

```bash
git add assets/
git commit -m "assets: add forklift, pallet-asm, pallet, cone, warning-sign USDs for layout v2"
```

---

## Task 2: Extend explore_rack.py to print X/Y footprint

Needed to set `ISLAND_RACK_DX` so the 2 racks in an island don't overlap.

**Files:**
- Modify: `asset_sandbox/scripts/explore_rack.py` (in `_print_rack_bounds`)

- [ ] **Step 1: Add X/Y bbox output**

In `_print_rack_bounds`, replace the bbox-compute block (the part from `box = bbox.GetBox()` through the `[OK]/[ACTION]` prints) with:

```python
        box = bbox.GetBox()
        mn, mx = box.GetMin(), box.GetMax()
        meters_per_unit = 0.01
        min_x_m, max_x_m = mn[0] * meters_per_unit, mx[0] * meters_per_unit
        min_y_m, max_y_m = mn[1] * meters_per_unit, mx[1] * meters_per_unit
        min_z_m, max_z_m = mn[2] * meters_per_unit, mx[2] * meters_per_unit
        footprint_x = max_x_m - min_x_m
        footprint_y = max_y_m - min_y_m
        estimated = round(max_z_m * 0.75, 2)
        print(f"[BOUNDS] X (m): min={min_x_m:.3f} max={max_x_m:.3f}  footprint_x={footprint_x:.3f}")
        print(f"[BOUNDS] Y (m): min={min_y_m:.3f} max={max_y_m:.3f}  footprint_y={footprint_y:.3f}")
        print(f"[BOUNDS] Z (m): min={min_z_m:.3f} max={max_z_m:.3f}")
        print(f"[SHELF]  Estimated top-shelf Z: ~{estimated} m")
        print(f"[CURRENT] warehouse_scene.RACK_SHELF_Z = {RACK_SHELF_Z}")
        print(f"[ACTION] Set ISLAND_RACK_DX > footprint_x ({footprint_x:.2f} m) so island racks don't overlap")
```

- [ ] **Step 2: USER runs the script**

Run:
```bash
conda activate isaaclab
python asset_sandbox/scripts/explore_rack.py
```
Expected: prints `footprint_x` and `footprint_y` in meters. **Record footprint_x** — sets ISLAND_RACK_DX in Task 4 (use `footprint_x + 0.3` for a safe gap; if unknown, default 1.5).

- [ ] **Step 3: Commit**

```bash
git add asset_sandbox/scripts/explore_rack.py
git commit -m "tooling: print rack X/Y footprint for island spacing"
```

---

## Task 3: Pure layout-grid module (TDD)

**Files:**
- Create: `env/layout_grid.py`
- Test: `tests/test_layout_grid.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_layout_grid.py`:

```python
# Pure-python layout math tests (no Isaac Sim needed). Run: pytest tests/test_layout_grid.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.layout_grid import island_rack_positions, item_specs


def test_island_rack_count():
    racks = island_rack_positions((-6.0, 0.0, 6.0), (8.0, 1.0, -5.0), 1.5)
    assert len(racks) == 18  # 9 islands x 2 racks


def test_island_rack_offsets():
    racks = island_rack_positions((0.0,), (0.0,), 1.5)
    assert racks == [(-0.75, 0.0, 0.0), (0.75, 0.0, 0.0)]


def test_item_specs_count_and_cycle():
    racks = island_rack_positions((-6.0, 0.0, 6.0), (8.0, 1.0, -5.0), 1.5)
    items = item_specs(racks, (0.21, 0.32, 0.52), 1.5)
    assert len(items) == 18
    assert items[0][0] == "fragile_0"
    assert items[1][0] == "regular_0"
    assert items[2][0] == "heavy_0"
    assert items[3][0] == "fragile_1"


def test_item_z_sits_on_shelf():
    items = item_specs([(0.0, 0.0, 0.0)], (0.21, 0.32, 0.52), 1.5)
    name, size, pos = items[0]
    assert size == 0.21
    assert abs(pos[2] - (1.5 + 0.105)) < 1e-9
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_layout_grid.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'env.layout_grid'`

- [ ] **Step 3: Implement the module**

Create `env/layout_grid.py`:

```python
# layout_grid.py
# Pure-python warehouse layout math. NO isaaclab import -> unit-testable standalone.
"""Grid generators for rack-island positions and item placement."""

from __future__ import annotations

_CATEGORIES = ("fragile", "regular", "heavy")


def island_rack_positions(
    cols_x: tuple[float, ...],
    rows_y: tuple[float, ...],
    rack_dx: float,
) -> list[tuple[float, float, float]]:
    """Return (x, y, z=0) for 2 racks per island over a cols_x x rows_y grid.

    Each island center (cx, cy) yields racks at (cx - rack_dx/2, cy) and
    (cx + rack_dx/2, cy). Row-major: all islands in rows_y[0] first.
    """
    out: list[tuple[float, float, float]] = []
    for cy in rows_y:
        for cx in cols_x:
            out.append((cx - rack_dx / 2.0, cy, 0.0))
            out.append((cx + rack_dx / 2.0, cy, 0.0))
    return out


def item_specs(
    rack_positions: list[tuple[float, float, float]],
    sizes: tuple[float, float, float],
    shelf_z: float,
) -> list[tuple[str, float, tuple[float, float, float]]]:
    """One box per rack, cycling category by index (fragile/regular/heavy).

    sizes = (fragile, regular, heavy) edge lengths in meters.
    Returns (name, size, (x, y, shelf_z + size/2)).
    """
    counters = {c: 0 for c in _CATEGORIES}
    out: list[tuple[str, float, tuple[float, float, float]]] = []
    for i, (x, y, _z) in enumerate(rack_positions):
        cat = _CATEGORIES[i % 3]
        size = sizes[i % 3]
        name = f"{cat}_{counters[cat]}"
        counters[cat] += 1
        out.append((name, size, (x, y, shelf_z + size / 2.0)))
    return out
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_layout_grid.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add env/layout_grid.py tests/test_layout_grid.py
git commit -m "feat: pure-python rack-island + item grid generators (TDD)"
```

---

## Task 4: Rectangular room + island layout in warehouse_scene.py

**Files:**
- Modify: `env/warehouse_scene.py`

- [ ] **Step 1: Rectangular room constants**

Replace the room geometry block:

```python
# ── Room Geometry ─────────────────────────────────────────────────────
ROOM_HALF_X = 10.0   # room width 20 m (x in [-10, +10])
ROOM_HALF_Y = 15.0   # room depth 30 m (y in [-15, +15])
WALL_H      = 6.0
WALL_T      = 0.3
WALL_COLOR  = (0.82, 0.81, 0.78)
```

- [ ] **Step 2: Island layout constants (use layout_grid)**

Add import near top (with other `from env...` imports):

```python
from env.layout_grid import island_rack_positions, item_specs
```

Replace the `RACK_POSITIONS` / `RACK_SHELF_Z` / `ITEM_SPECS` block:

```python
# Island grid: 3 columns x 3 rows = 9 islands, 2 racks each = 18 racks.
ISLAND_COLS_X = (-6.0, 0.0, 6.0)
ISLAND_ROWS_Y = (8.0, 1.0, -5.0)
ISLAND_RACK_DX = 1.5   # intra-island rack spacing; MUST exceed rack footprint_x (Task 2)

RACK_POSITIONS = island_rack_positions(ISLAND_COLS_X, ISLAND_ROWS_Y, ISLAND_RACK_DX)

# RACK_SHELF_Z: measured 2026-05-30 via explore_rack.py (199.9 cm * 0.01 = 2.0 m rack; shelf @75% = 1.5 m)
RACK_SHELF_Z = 1.5

ITEM_SPECS = item_specs(
    RACK_POSITIONS,
    (BOX_SMALL_SIZE, BOX_MED_SIZE, BOX_LARGE_SIZE),
    RACK_SHELF_Z,
)
```

- [ ] **Step 3: Rectangular walls**

Replace the 4 wall declarations inside `WarehouseSceneCfg` (wall_n/s/e/w) to use ROOM_HALF_X/Y:

```python
    wall_n = _wall_cfg("wall_n", (0.0,  ROOM_HALF_Y, WALL_H / 2),
                       (ROOM_HALF_X * 2 + 2 * WALL_T, WALL_T, WALL_H))
    wall_s = _wall_cfg("wall_s", (0.0, -ROOM_HALF_Y, WALL_H / 2),
                       (ROOM_HALF_X * 2 + 2 * WALL_T, WALL_T, WALL_H))
    wall_e = _wall_cfg("wall_e", ( ROOM_HALF_X, 0.0, WALL_H / 2),
                       (WALL_T, ROOM_HALF_Y * 2, WALL_H))
    wall_w = _wall_cfg("wall_w", (-ROOM_HALF_X, 0.0, WALL_H / 2),
                       (WALL_T, ROOM_HALF_Y * 2, WALL_H))
```

- [ ] **Step 4: Commit**

```bash
git add env/warehouse_scene.py
git commit -m "feat: rectangular 20x30 room + 9-island rack grid"
```

---

## Task 5: Spawn racks/items/props via __post_init__

InteractiveScene reads `cfg.__dict__`, so adding AssetBaseCfg attributes in `__post_init__` (after dataclass init) registers them. This replaces the 6 explicit rack_N / item attributes (now 18+18+props would be unwieldy).

**Files:**
- Modify: `env/warehouse_scene.py`

- [ ] **Step 1: Add prop USD constants**

After `BOX_USD = {...}` add:

```python
FORKLIFT_USD = _usd("Equipment/Forklifts/Forklift_A/Forklift_A01_PR_V_NVD_01.usd")
PALLET_ASM_USD = [
    _usd("Shipping/Cardboard_Boxes_on_Pallet/Pallet_Asm_A/Pallet_Asm_A01_66x66x46cm_PR_V_NVD_01.usd"),
    _usd("Shipping/Cardboard_Boxes_on_Pallet/Pallet_Asm_A/Pallet_Asm_A04_120x122x75cm_PR_V_NVD_01.usd"),
]
PALLET_USD = _usd("Shipping/Pallets/Plastic/Economy_A/EconomyPlasticPallet_A01_PR_NVD_01.usd")
CONE_USD = _usd("Safety/Cones/Heavy-Duty_Traffic/HeavyDutyTrafficCone_A02_71cm_PR_V_NVD_01.usd")
SIGN_USD = _usd("Safety/Floor_Signs/Warning_A/WarningSign_A01_PR_NVD_01.usd")

# Static decorative/obstacle props: (name, usd_path, (x, y, z))
PROP_SPECS = [
    ("forklift_0",   FORKLIFT_USD,      (-5.0, 12.0, 0.0)),
    ("palletasm_0",  PALLET_ASM_USD[0], ( 3.0, 12.0, 0.0)),
    ("palletasm_1",  PALLET_ASM_USD[1], ( 5.0, 12.0, 0.0)),
    ("pallet_0",     PALLET_USD,        (-3.0, -7.0, 0.0)),
    ("pallet_1",     PALLET_USD,        ( 4.0, -7.0, 0.0)),
    ("cone_0",       CONE_USD,          (-3.0,  4.0, 0.0)),
    ("cone_1",       CONE_USD,          ( 3.0,  4.0, 0.0)),
    ("cone_2",       CONE_USD,          (-3.0, -2.0, 0.0)),
    ("cone_3",       CONE_USD,          ( 3.0, -2.0, 0.0)),
    ("sign_0",       SIGN_USD,          (-6.0, -9.5, 0.0)),
    ("sign_1",       SIGN_USD,          ( 6.0, -9.5, 0.0)),
]
```

- [ ] **Step 2: Add a generic static-USD prop factory**

Next to `_rack_cfg`:

```python
def _prop_cfg(name: str, usd_path: str, pos: tuple) -> AssetBaseCfg:
    """Static USD prop (cm->m scale, collision on). For forklift/pallets/cones/signs."""
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(0.01, 0.01, 0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )
```

- [ ] **Step 3: Remove explicit rack_0..rack_5 + item attributes, add __post_init__**

Delete the explicit `rack_0`..`rack_5`, `fragile_0`..`heavy_1` attribute lines from `WarehouseSceneCfg`. Keep `ground`, `dome_light`, `robot`, `camera`, walls, `zone_A/B/C`. Then add at the end of the class:

```python
    def __post_init__(self) -> None:
        """Spawn 18 racks + 18 items + props (too many for explicit attrs)."""
        for i, pos in enumerate(RACK_POSITIONS):
            setattr(self, f"rack_{i}", _rack_cfg(i, pos))
        for name, size, pos in ITEM_SPECS:
            setattr(self, name, _item_cfg(name, size, pos))
        for name, usd_path, pos in PROP_SPECS:
            setattr(self, name, _prop_cfg(name, usd_path, pos))
```

- [ ] **Step 4: USER verifies scene loads**

Run:
```bash
conda activate isaaclab
python asset_sandbox/scripts/explore_scene.py
```
Expected: window shows 18 racks in 3x3 islands, boxes on shelves, forklift + pallets in receiving (north), cones in aisles, signs near zones (south). Console prints layout summary with 18 items. No "prim not found" / fatal MDL errors.
If racks/items missing -> `__post_init__` setattr not registered: fallback to explicit attributes (declare rack_0..rack_17, item attrs by name).

- [ ] **Step 5: Commit**

```bash
git add env/warehouse_scene.py
git commit -m "feat: spawn 18 racks + 18 items + warehouse props via __post_init__"
```

---

## Task 6: Env params + receiving-north spawn

**Files:**
- Modify: `env/warehouse_env.py`

- [ ] **Step 1: Scene num_envs + spacing**

In `WarehouseEnvCfg`, change the scene field default:

```python
    scene: WarehouseSceneCfg = WarehouseSceneCfg(num_envs=2, env_spacing=32.0)
```

- [ ] **Step 2: Episode length**

In `WarehouseEnvCfg.__post_init__`, change:

```python
        self.episode_length_s = 60.0   # 60s x 10Hz = 600 steps; ~25m traverse @1m/s + nav
```

- [ ] **Step 3: Receiving-north spawn**

In `EventCfg.reset_robot`, change pose_range:

```python
            # Receiving area (north); robot must navigate south through islands to a zone.
            "pose_range": {"x": (-8.0, 8.0), "y": (11.0, 14.0), "yaw": (-3.14, 3.14)},
```

- [ ] **Step 4: USER verifies env builds + obs contract**

Run:
```bash
conda activate isaaclab
python tests/test_env.py --num_envs 1
```
Expected: all PASS — obs keys pixels/position/goal/goal_emb with correct shapes, action_space Box(-1,1,(2,)), 10 steps run, reward shape (1,).

- [ ] **Step 5: Commit**

```bash
git add env/warehouse_env.py
git commit -m "feat: 20x30 env params (num_envs=2, spacing=32, episode=60s, receiving spawn)"
```

---

## Task 7: Rectangular out_of_bounds

**Files:**
- Modify: `env/warehouse_reward.py`
- Modify: `env/warehouse_env.py` (TerminationsCfg params)

- [ ] **Step 1: Rectangular bounds in reward fn**

Replace `out_of_bounds` in `warehouse_reward.py`:

```python
def out_of_bounds(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    half_extent_x: float = 9.5,
    half_extent_y: float = 14.5,
) -> torch.Tensor:
    """Termination: True if robot leaves the rectangular room interior."""
    xy = _robot_xy(env, asset_cfg)
    out_x = xy[:, 0].abs() > half_extent_x
    out_y = xy[:, 1].abs() > half_extent_y
    return out_x | out_y
```

- [ ] **Step 2: Pass bounds from TerminationsCfg**

In `warehouse_env.py` `TerminationsCfg`, change the `bounds` term to pass params:

```python
    bounds = DoneTerm(
        func=out_of_bounds,
        params={"half_extent_x": 9.5, "half_extent_y": 14.5},
    )
```

- [ ] **Step 3: USER re-runs contract test**

Run: `python tests/test_env.py --num_envs 1`
Expected: all PASS (no termination regressions).

- [ ] **Step 4: Commit**

```bash
git add env/warehouse_reward.py env/warehouse_env.py
git commit -m "feat: rectangular out_of_bounds (x +-9.5, y +-14.5)"
```

---

## Task 8: Mirror config yaml

**Files:**
- Modify: `configs/env_config.yaml`

- [ ] **Step 1: Update env + scene sections**

Set: `num_envs: 2`, `env_spacing: 32.0`, `episode_length_s: 60.0`. Under `scene.room` replace `half_extent: 10.0` with `half_extent_x: 10.0` and `half_extent_y: 15.0`. Under `termination` set `out_of_bounds_half_extent_x: 9.5`, `out_of_bounds_half_extent_y: 14.5`. Under `scene.racks` set `count: 18` and add `islands: "3x3 grid, 2 racks each"`, `island_cols_x: [-6, 0, 6]`, `island_rows_y: [8, 1, -5]`, `island_rack_dx: 1.5`. Under `scene.items` set `count: 18`. Add a `scene.props` block:

```yaml
  props:
    forklift: { count: 1, usd: "Equipment/Forklifts/Forklift_A/Forklift_A01_PR_V_NVD_01.usd" }
    boxes_on_pallet: { count: 2, usd: "Shipping/Cardboard_Boxes_on_Pallet/Pallet_Asm_A/" }
    empty_pallet: { count: 2, usd: "Shipping/Pallets/Plastic/Economy_A/EconomyPlasticPallet_A01_PR_NVD_01.usd" }
    cone: { count: 4, usd: "Safety/Cones/Heavy-Duty_Traffic/HeavyDutyTrafficCone_A02_71cm_PR_V_NVD_01.usd" }
    warning_sign: { count: 2, usd: "Safety/Floor_Signs/Warning_A/WarningSign_A01_PR_NVD_01.usd" }
```

- [ ] **Step 2: Update spawn note**

Under `robot` set `spawn_range_xy` note to receiving-north: add `spawn_x: [-8, 8]`, `spawn_y: [11, 14]`.

- [ ] **Step 3: Commit**

```bash
git add configs/env_config.yaml
git commit -m "docs: mirror layout v2 params in env_config.yaml"
```

---

## Task 9: Integration verification (USER-run)

**Files:** none (verification only; commit any tuning)

- [ ] **Step 1: Visual scene check**

Run: `python asset_sandbox/scripts/explore_scene.py`
Expected: realistic warehouse — 3x3 rack islands with boxes, receiving props north, zones+signs south, cones in aisles. No floating/overlapping racks. If racks overlap -> raise `ISLAND_RACK_DX` (Task 4 Step 2); if boxes float/clip -> tune `RACK_SHELF_Z`.

- [ ] **Step 2: Contract test**

Run: `python tests/test_env.py --num_envs 1`
Expected: ALL PASS.

- [ ] **Step 3: VRAM / OOM test at num_envs=2**

Run: `python scripts/run_env.py --num_envs 2 --headless --steps 200`
Expected: runs 200 steps, prints reward stats, no CUDA OOM. Watch VRAM (`nvidia-smi`). If stable and < ~6.5 GB -> optionally try `--num_envs 4`.

- [ ] **Step 4: Pure unit tests still green**

Run: `pytest tests/test_layout_grid.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit any tuning + update docs**

```bash
git add -A
git commit -m "chore: tune ISLAND_RACK_DX / RACK_SHELF_Z after visual verification"
```
Then update `docs/specs/environment.md` layout section to match v2 (room 20x30, 9 islands, props, num_envs 2, episode 60s).

---

## Self-Review Notes

- **Spec coverage:** room 20x30 (T4), env_spacing 32 (T6), 9 islands/18 racks (T3,T4,T5), 18 boxes (T3,T5), props all 5 types (T1,T5), receiving spawn (T6), rectangular out_of_bounds (T7), num_envs 2 / episode 60 (T6), yaml mirror (T8), rack footprint measure (T2), verify plan (T9). All covered.
- **Mission unchanged:** obs contract + zones untouched; verified by test_env (T6,T9).
- **cm->m scale 0.01:** applied to props (T5 `_prop_cfg`) and already on racks/items.
- **Risk:** `__post_init__` setattr registration (T5) — fallback to explicit attrs documented in T5 Step 4.
- **Out of scope (unchanged):** reward rebalance/collision/curriculum (Pass 3-5), building shell, pickup.
