# Box Physics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply rigid body physics and category-specific masses to cardboard boxes so they act as dynamic physical objects instead of static floating colliders.

**Architecture:** We will modify `env/layout_grid.py` to route mass logic (`(2.0, 6.0, 12.0)`) into the item specifications alongside their sizes. Then in `env/warehouse_scene.py`, we unpack this mass, enrich the Isaac Lab `UsdFileCfg` with `RigidBodyPropertiesCfg`, `MassPropertiesCfg(mass=mass)`, and high-friction/zero-restitution `RigidBodyMaterialCfg` so they fall and rest on the warehouse floor/racks realistically.

**Tech Stack:** Python, Isaac Lab (`sim_utils`)

---

### Task 1: Update Layout Grid Logic and Tests

**Files:**
- Modify: `tests/test_layout_grid.py:20-35`
- Modify: `env/layout_grid.py:28-47`

- [ ] **Step 1: Write the failing tests**

Update `tests/test_layout_grid.py` to pass the `masses` tuple and check for the `mass` field in the return tuple:

```python
def test_item_specs_count_and_cycle():
    racks = island_rack_positions((-6.0, 0.0, 6.0), (8.0, 1.0, -5.0), 1.5)
    items = item_specs(racks, (0.21, 0.32, 0.52), (2.0, 6.0, 12.0), 1.5)
    assert len(items) == 18
    assert items[0][0] == "fragile_0"
    assert items[0][2] == 2.0  # mass
    assert items[1][0] == "regular_0"
    assert items[1][2] == 6.0
    assert items[2][0] == "heavy_0"
    assert items[2][2] == 12.0
    assert items[3][0] == "fragile_1"


def test_item_z_sits_on_shelf():
    items = item_specs([(0.0, 0.0, 0.0)], (0.21, 0.32, 0.52), (2.0, 6.0, 12.0), 1.5)
    name, size, mass, pos = items[0]
    assert size == 0.21
    assert mass == 2.0
    assert abs(pos[2] - (1.5 + 0.105)) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_layout_grid.py -v`
Expected: FAIL because `item_specs` does not expect the new `masses` argument.

- [ ] **Step 3: Write minimal implementation**

Update `env/layout_grid.py` to accept `masses` and output it in the tuple.

```python
def item_specs(
    rack_positions: list[tuple[float, float, float]],
    sizes: tuple[float, float, float],
    masses: tuple[float, float, float],
    shelf_z: float,
) -> list[tuple[str, float, float, tuple[float, float, float]]]:
    """One box per rack, cycling category by index (fragile/regular/heavy).

    sizes = (fragile_m, regular_m, heavy_m) edge lengths in meters.
    masses = (fragile_kg, regular_kg, heavy_kg) weights in kg.
    Returns list of (name, size, mass, (x, y, shelf_z + size/2)).
    """
    counters = {c: 0 for c in _CATEGORIES}
    out: list[tuple[str, float, float, tuple[float, float, float]]] = []
    for i, (x, y, _z) in enumerate(rack_positions):
        cat = _CATEGORIES[i % 3]
        size = sizes[i % 3]
        mass = masses[i % 3]
        name = f"{cat}_{counters[cat]}"
        counters[cat] += 1
        out.append((name, size, mass, (x, y, shelf_z + size / 2.0)))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_layout_grid.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add env/layout_grid.py tests/test_layout_grid.py
git commit -m "test: update layout grid item_specs to generate mass"
```

### Task 2: Apply Rigid Body Properties in Scene

**Files:**
- Modify: `env/warehouse_scene.py`

- [ ] **Step 1: Write the implementation**

In `env/warehouse_scene.py`, modify constants and configs. Add `BOX_MASSES`, update `ITEM_SPECS`, modify `_item_cfg`, and fix the loop in `__post_init__`.

First, find `BOX_LARGE_SIZE` (around line 67) and add the new constant:
```python
BOX_LARGE_SIZE = 0.52  # heavy
BOX_MASSES = (2.0, 6.0, 12.0)  # fragile, regular, heavy
```

Update `ITEM_SPECS` generation (around line 139):
```python
ITEM_SPECS = _item_specs_gen(
    RACK_POSITIONS,
    (BOX_SMALL_SIZE, BOX_MED_SIZE, BOX_LARGE_SIZE),
    BOX_MASSES,
    RACK_SHELF_Z,
)
```

Update `_item_cfg` (around line 195):
```python
def _item_cfg(name: str, size: float, mass: float, pos: tuple) -> AssetBaseCfg:
    """Static cardboard-box USD (size encodes category for CLIP; cm-authored -> scale 0.01)."""
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BOX_USD[size],
            scale=(0.01, 0.01, 0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.0
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )
```

Update the unpack loop in `__post_init__` (around line 298):
```python
        for name, size, mass, pos in ITEM_SPECS:
            setattr(self, name, _item_cfg(name, size, mass, pos))
```

- [ ] **Step 2: Verify code syntax passes tests**

Run: `pytest tests/test_env.py --num_envs 1 -v`
Expected: PASS (or loads warehouse app up successfully without crashing due to config mismatch).

- [ ] **Step 3: Commit**

```bash
git add env/warehouse_scene.py
git commit -m "feat: apply rigid body properties and mass to boxes in scene"
```
