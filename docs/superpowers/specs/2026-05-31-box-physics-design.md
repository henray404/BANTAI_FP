# Warehouse Box Physics Design

## Overview
The goal of this change is to apply rigid body physics to the cardboard box assets in the warehouse simulation. Currently, the boxes act as static colliders and float in mid-air. We need them to be dynamic bodies that fall under gravity, with realistic cardboard friction and categorized masses (2kg, 6kg, 12kg) based on their size (fragile, regular, heavy).

## Architecture (Approach B)
We will adopt a modular approach, separating the definition of physical properties (mass) in the layout logic from the application of those properties in the 3D scene configuration.

### 1. Data Logic (`env/layout_grid.py`)
- **Modification**: Update the `item_specs` generator function.
- **Input**: Add a new parameter `masses: tuple[float, float, float]` alongside the existing `sizes`.
- **Output**: The function will now return a list of tuples containing four elements: `(name, size, mass, pos)`. The mass will cycle through the `masses` tuple corresponding to the `_CATEGORIES`.

### 2. Scene Configuration (`env/warehouse_scene.py`)
- **Constants**: Define a new constant `BOX_MASSES = (2.0, 6.0, 12.0)`.
- **Generation**: Pass `BOX_MASSES` to the `item_specs` function call when creating the `ITEM_SPECS` constant.
- **Factory Update**: Modify the `_item_cfg` factory function signature to accept the new mass parameter: `def _item_cfg(name: str, size: float, mass: float, pos: tuple) -> AssetBaseCfg:`.
- **Physics Implementation**: Within `_item_cfg`, enrich the `UsdFileCfg` with:
  - `rigid_props=sim_utils.RigidBodyPropertiesCfg()` to enable dynamic physics (gravity).
  - `mass_props=sim_utils.MassPropertiesCfg(mass=mass)` to apply the specific category weight.
  - Define a physics material on the `UsdFileCfg` (or via collision properties if applicable) using `sim_utils.RigidBodyMaterialCfg` with high friction (`static_friction=0.8`, `dynamic_friction=0.6`) and zero bounciness (`restitution=0.0`) to simulate a cardboard box on the floor.
- **Loop Update**: Update the `__post_init__` loop that iterates over `ITEM_SPECS` to unpack the new `mass` value and pass it to `_item_cfg`.
