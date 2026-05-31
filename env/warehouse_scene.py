# warehouse_scene.py
# Person 1 — Warehouse scene configuration
#
# IMPORT RULE: No AppLauncher here. Imported by warehouse_env.py and entry scripts.
# See bugs_errors/2026-05-15_double-applaunch-crash.md.
#
# USD ASSET POLICY: External USD assets approved by user (2026-05-22).
# See bugs_errors/2026-05-22_usd-assets-size-categories.md.

"""Warehouse scene config — Layout v2.

Room: 20 x 30 m (x in [-10,+10], y in [-15,+15]).

    NORTH WALL  (y = +15)
    ===== RECEIVING (y +11..+14) =====
    [forklift]  [pallet-asm]  [pallet-asm]    <- robot spawns here
    --------------------------------------------------
    [I1 rk rk]  [I2 rk rk]  [I3 rk rk]   y=+8  island row 1
    [I4 rk rk]  [I5 rk rk]  [I6 rk rk]   y=+1  island row 2
    [I7 rk rk]  [I8 rk rk]  [I9 rk rk]   y=-5  island row 3
     x=-6        x=0          x=+6
        [empty-pallet]    [cone][cone]
    ===== SHIPPING (y -9..-14) =======
    [ZONE_A]   [ZONE_B]   [ZONE_C]         y=-12
     orange      cyan      purple
   (fragile)  (regular)  (heavy)
    SOUTH WALL  (y = -15)

9 islands (3x3), 2 racks each = 18 racks total.
Category encoding:
    fragile = CubeBox 21 cm   zone_A (orange)
    regular = CubeBox 32 cm   zone_B (cyan)
    heavy   = CubeBox 52 cm   zone_C (purple)
USD scale: scale=(0.01,0.01,0.01) on all NVIDIA DT assets (cm-authored, no auto-convert).
Robot: Jetbot (Nucleus USD). Spawns receiving-north.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from env.layout_grid import island_rack_positions, item_specs as _item_specs_gen


# ── Asset Paths ───────────────────────────────────────────────────────
ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"


def _usd(rel: str) -> str:
    """Convert relative asset path to forward-slash string for Isaac Lab."""
    return str(ASSETS_DIR / rel).replace("\\", "/")


RACK_USD = _usd("Shelving/Racks/Rack_L/Rack_L01_PR_NVD_01.usd")

# Item box sizes (meters) — size encodes category for CLIP perception.
BOX_SMALL_SIZE = 0.21  # fragile
BOX_MED_SIZE   = 0.32  # regular
BOX_LARGE_SIZE = 0.52  # heavy
BOX_MASSES     = (2.0, 6.0, 12.0)  # fragile, regular, heavy

BOX_USD = {
    BOX_SMALL_SIZE: _usd("Shipping/Cardboard_Boxes/Cube_A/CubeBox_A03_21cm_PR_NVD_01.usd"),
    BOX_MED_SIZE:   _usd("Shipping/Cardboard_Boxes/Cube_A/CubeBox_A05_32cm_PR_NVD_01.usd"),
    BOX_LARGE_SIZE: _usd("Shipping/Cardboard_Boxes/Cube_A/CubeBox_A07_52cm_PR_NVD_01.usd"),
}

# Zone -> item category mapping (for text instruction generation)
ZONE_ITEM_MAP: dict[str, str] = {
    "zone_A": "fragile",
    "zone_B": "regular",
    "zone_C": "heavy",
}

# Prop USD paths (all cm-authored, scale 0.01 applied in _prop_cfg)
FORKLIFT_USD   = _usd("Equipment/Forklifts/Forklift_A/Forklift_A01_PR_V_NVD_01.usd")
PALLET_ASM_USD = [
    _usd("Shipping/Cardboard_Boxes_on_Pallet/Pallet_Asm_A/Pallet_Asm_A01_66x66x46cm_PR_V_NVD_01.usd"),
    _usd("Shipping/Cardboard_Boxes_on_Pallet/Pallet_Asm_A/Pallet_Asm_A04_120x122x75cm_PR_V_NVD_01.usd"),
]
PALLET_USD = _usd("Shipping/Pallets/Plastic/Economy_A/EconomyPlasticPallet_A01_PR_NVD_01.usd")
CONE_USD   = _usd("Safety/Cones/Heavy-Duty_Traffic/HeavyDutyTrafficCone_A02_71cm_PR_V_NVD_01.usd")
SIGN_USD   = _usd("Safety/Floor_Signs/Warning_A/WarningSign_A01_PR_NVD_01.usd")


# ── Robot Config ──────────────────────────────────────────────────────
JETBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=5.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),
    ),
    actuators={
        "wheel_acts": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            damping=None,
            stiffness=None,
            effort_limit_sim=100.0,
            velocity_limit_sim=50.0,
        ),
    },
)


# ── Room Geometry ─────────────────────────────────────────────────────
ROOM_HALF_X = 10.0   # room width 20 m (x in [-10, +10])
ROOM_HALF_Y = 15.0   # room depth 30 m (y in [-15, +15])
WALL_H      = 6.0
WALL_T      = 0.3
WALL_COLOR  = (0.82, 0.81, 0.78)


# ── Scene Layout Constants ────────────────────────────────────────────
# 9-island grid (3 columns x 3 rows), 2 racks per island = 18 racks.
# ISLAND_RACK_DX: intra-island x spacing between 2 racks.
# Set > rack footprint_x (run explore_rack.py to measure). Default 1.5 m.
ISLAND_COLS_X  = (-6.0, 0.0, 6.0)
ISLAND_ROWS_Y  = (8.0, 1.0, -5.0)
ISLAND_RACK_DX = 1.5

RACK_POSITIONS = island_rack_positions(ISLAND_COLS_X, ISLAND_ROWS_Y, ISLAND_RACK_DX)

# RACK_SHELF_Z: measured 2026-05-30 via explore_rack.py.
#   raw bbox max_z = 199.902 (USD cm units) -> 199.902 * 0.01 = ~2.0 m rack in sim.
#   top shelf @ 75%: 149.93 * 0.01 = 1.5 m
RACK_SHELF_Z = 1.5

ITEM_SPECS = _item_specs_gen(
    RACK_POSITIONS,
    (BOX_SMALL_SIZE, BOX_MED_SIZE, BOX_LARGE_SIZE),
    BOX_MASSES,
    RACK_SHELF_Z,
)

ZONE_SIZE  = (3.0, 3.0, 0.02)
ZONE_SPECS = [
    ("zone_A", (1.0, 0.9, 0.0), (-6.0, -12.0, 0.01)),  # orange — fragile
    ("zone_B", (0.0, 0.9, 0.9), ( 0.0, -12.0, 0.01)),  # cyan   — regular
    ("zone_C", (0.7, 0.0, 0.9), ( 6.0, -12.0, 0.01)),  # purple — heavy
]

# Static props: (name, usd_path, (x, y, z))
PROP_SPECS = [
    ("forklift_0",  FORKLIFT_USD,      (-5.0,  12.0, 0.0)),
    ("palletasm_0", PALLET_ASM_USD[0], ( 3.0,  12.0, 0.0)),
    ("palletasm_1", PALLET_ASM_USD[1], ( 5.0,  12.0, 0.0)),
    ("pallet_0",    PALLET_USD,        (-3.0,  -7.0, 0.0)),
    ("pallet_1",    PALLET_USD,        ( 4.0,  -7.0, 0.0)),
    ("cone_0",      CONE_USD,          (-3.0,   4.0, 0.0)),
    ("cone_1",      CONE_USD,          ( 3.0,   4.0, 0.0)),
    ("cone_2",      CONE_USD,          (-3.0,  -2.0, 0.0)),
    ("cone_3",      CONE_USD,          ( 3.0,  -2.0, 0.0)),
    ("sign_0",      SIGN_USD,          (-6.0,  -9.5, 0.0)),
    ("sign_1",      SIGN_USD,          ( 6.0,  -9.5, 0.0)),
]


# ── Scene Element Factories ───────────────────────────────────────────
def _wall_cfg(name: str, pos: tuple, size: tuple) -> AssetBaseCfg:
    """Primitive concrete-grey wall panel."""
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=WALL_COLOR),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _rack_cfg(idx: int, pos: tuple) -> AssetBaseCfg:
    """Static Rack_L01 USD (cm-authored -> scale 0.01; verify height via explore_rack.py)."""
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Rack_{idx}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=RACK_USD,
            scale=(0.01, 0.01, 0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _item_cfg(name: str, size: float, pos: tuple) -> AssetBaseCfg:
    """Static cardboard-box USD (size encodes category for CLIP; cm-authored -> scale 0.01)."""
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BOX_USD[size],
            scale=(0.01, 0.01, 0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _prop_cfg(name: str, usd_path: str, pos: tuple) -> AssetBaseCfg:
    """Static USD prop (cm-authored, scale 0.01, with collision)."""
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(0.01, 0.01, 0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _zone_cfg(name: str, color: tuple, pos: tuple) -> AssetBaseCfg:
    """Flat colored delivery zone slab."""
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=ZONE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


# ── Scene Cfg ─────────────────────────────────────────────────────────
@configclass
class WarehouseSceneCfg(InteractiveSceneCfg):
    """20 x 30 m warehouse: 9 rack islands, receiving north, shipping south."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.0,
            ),
        ),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)),
    )

    robot: ArticulationCfg = JETBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/onboard_cam",
        update_period=0.1,
        height=64,
        width=64,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 50.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.10),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )

    # Walls (rectangular 20 x 30 m)
    wall_n = _wall_cfg("wall_n", (0.0,  ROOM_HALF_Y, WALL_H / 2),
                       (ROOM_HALF_X * 2 + 2 * WALL_T, WALL_T, WALL_H))
    wall_s = _wall_cfg("wall_s", (0.0, -ROOM_HALF_Y, WALL_H / 2),
                       (ROOM_HALF_X * 2 + 2 * WALL_T, WALL_T, WALL_H))
    wall_e = _wall_cfg("wall_e", ( ROOM_HALF_X, 0.0, WALL_H / 2),
                       (WALL_T, ROOM_HALF_Y * 2, WALL_H))
    wall_w = _wall_cfg("wall_w", (-ROOM_HALF_X, 0.0, WALL_H / 2),
                       (WALL_T, ROOM_HALF_Y * 2, WALL_H))

    # Delivery zones (colored goal slabs — shipping south)
    zone_A = _zone_cfg(*ZONE_SPECS[0])
    zone_B = _zone_cfg(*ZONE_SPECS[1])
    zone_C = _zone_cfg(*ZONE_SPECS[2])

    def __post_init__(self) -> None:
        """Register 18 racks + 18 items + 11 props as scene attributes.

        InteractiveSceneCfg picks up any AssetBaseCfg set on self after __post_init__.
        Using setattr avoids declaring 47 explicit class-level attributes.
        """
        for i, pos in enumerate(RACK_POSITIONS):
            setattr(self, f"rack_{i}", _rack_cfg(i, pos))
        for name, size, mass, pos in ITEM_SPECS:
            setattr(self, name, _item_cfg(name, size, mass, pos))
        for name, usd_path, pos in PROP_SPECS:
            setattr(self, name, _prop_cfg(name, usd_path, pos))
