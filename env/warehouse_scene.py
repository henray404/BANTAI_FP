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
54 boxes: 18 racks × 3 shelf levels, category cycles fragile/regular/heavy per rack.
Category encoding:
    fragile = CubeBox 21 cm   zone_A (orange)
    regular = CubeBox 32 cm   zone_B (cyan)
    heavy   = CubeBox 52 cm   zone_C (purple)
USD scale: scale=(0.01,0.01,0.01) on all NVIDIA DT assets (cm-authored, no auto-convert).
Robot: Ridgeback-Franka (Nucleus USD). Spawns receiving-north.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from env.layout_grid import (
    RACK_POSITIONS as _RACK_POSITIONS_LG,
    TARGET_BOX_SPECS as _TARGET_BOX_SPECS_LG,
    island_rack_positions,
)


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
PALLET_USD = _usd("Shipping/Pallets/Plastic/Economy_A/EconomyPlasticPallet_A01_PR_NVD_01.usd")
CONE_USD   = _usd("Safety/Cones/Heavy-Duty_Traffic/HeavyDutyTrafficCone_A02_71cm_PR_V_NVD_01.usd")
SIGN_USD   = _usd("Safety/Floor_Signs/Warning_A/WarningSign_A01_PR_NVD_01.usd")


# ── Robot Config ──────────────────────────────────────────────────────
# Ridgeback-Franka mobile manipulator: Clearpath Ridgeback (holonomic base) + Franka Panda
# 7-DOF arm + 2-finger gripper. Single pre-rigged articulation on the Isaac 5.1 server
# (HTTP 200 verified). Values mirror Isaac Lab's shipped RIDGEBACK_FRANKA_PANDA_CFG, with two
# project overrides: activate_contact_sensors=True (shipped False — collision sensor needs it)
# and raised solver iters for arm/contact stability. See docs/.../2026-06-01-arm-pickup-design.md.
#
# Base is HOLONOMIC via 3 dummy joints (prismatic_x/y + revolute_z), velocity-controlled.
# Driven diff-drive-style from a (2,) action (see warehouse_env.ActionsCfg / _base_cmd) so the
# obs/action contract is unchanged. Arm + gripper hold their tucked init pose (position control);
# the pickup state machine will script them later.
# VERIFY on first run: base chassis BODY name (contact sensor) + dummy-joint chain frame
# (if the base drives in a fixed world direction regardless of heading, _base_cmd must project
# by yaw — see its comment).
#
# CRITICAL (2026-06-09): the Isaac 5.1 ridgeback_franka.usd ships NO base anchor, so the
# articulation loads FLOATING (PhysX auto-roots at panda_link2). Driving the dummy holonomic
# joints then never translates the chassis in world space — the `world` leaf link absorbs the
# motion instead. We weld the `world` link to the stage world frame with a fixed joint at spawn
# (see _spawn_ridgeback_welded below) so it becomes fixed-base, re-roots at `world`, and the base
# drives correctly. Proof: base_link disp 0.00m -> 0.91m after weld. See
# bugs_errors/2026-06-09_ridgeback-floating-base.md.
def _weld_robot_world_links() -> None:
    """Anchor every robot `world` link to the stage world frame with a fixed joint.

    Scans the stage (handles 1..N cloned envs) and welds each `<...>/Robot/world` link so the
    floating Ridgeback articulation becomes fixed-base. Idempotent. Run after the USD spawn,
    before the physics view is created (i.e. before sim.reset()).
    """
    import omni.usd
    from pxr import Sdf, Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        if prim.GetName() != "world" or prim.GetParent().GetName() != "Robot":
            continue
        joint_path = Sdf.Path(f"{prim.GetParent().GetPath()}/base_world_anchor")
        if stage.GetPrimAtPath(joint_path).IsValid():
            continue  # already welded
        fixed = UsdPhysics.FixedJoint.Define(stage, joint_path)
        fixed.CreateBody1Rel().SetTargets([prim.GetPath()])  # body0 empty = stage world frame


def _spawn_ridgeback_welded(prim_path, cfg, translation=None, orientation=None):
    """Spawn the Ridgeback USD, then weld its `world` link (fix Isaac 5.1 floating base)."""
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    prim = spawn_from_usd(prim_path, cfg, translation, orientation)
    _weld_robot_world_links()
    return prim


RIDGEBACK_FRANKA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd",
        activate_contact_sensors=True,
        # Disable gravity on the arm so the DifferentialIK (relative mode) holds its pose
        # instead of sagging to a gravity-rest pose each step. Matches FRANKA_PANDA_HIGH_PD_CFG
        # (Isaac-Lift-Cube/Reach-Franka). Boxes keep gravity, so a held box still has weight.
        # See bugs_errors/2026-06-16_arm-sag-gravity-relative-ik.md.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # holonomic base (planar floating joint)
            "dummy_base_prismatic_y_joint": 0.0,
            "dummy_base_prismatic_x_joint": 0.0,
            "dummy_base_revolute_z_joint": 0.0,
            # franka arm — tucked/ready pose
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 2.0,
            "panda_joint7": 0.741,
            # gripper open
            "panda_finger_joint.*": 0.035,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=["dummy_base_.*"],
            effort_limit_sim=100000.0,
            stiffness=0.0,
            damping=1e5,
        ),
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=200.0,
            stiffness=1e5,
            damping=1e3,
        ),
    },
)

# Override the USD spawner with the welding wrapper so the floating base is anchored at spawn
# (Isaac 5.1 ridgeback_franka.usd has no base anchor — see _spawn_ridgeback_welded above).
RIDGEBACK_FRANKA_CFG.spawn.func = _spawn_ridgeback_welded


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

# RACK_POSITIONS sourced from layout_grid (pure-Python, no Isaac import) so tests can
# verify positions without launching Isaac Sim.
RACK_POSITIONS = _RACK_POSITIONS_LG

# Hardcoded shelf deck levels (top surface z, meters).
# Rack height ~2.0m (measured 2026-05-30). 3 evenly-spaced levels.
# Solid CuboidCfg shelf decks are spawned at these heights via _shelf_deck_cfg().
# Boxes fall from z=2.8m and land on the top deck (RACK_SHELF_LEVELS[-1]).
# Tune these values if boxes miss the deck: RACK_SHELF_LEVELS[2] should match actual top shelf.
RACK_SHELF_LEVELS = (0.72351, 1.32528, 1.92566)  # measured shelf surface z (bottom/mid/top)

# Shelf deck geometry — covers interior of rack frame without hitting uprights.
# 0.70m × 0.70m square = orientation-agnostic (rack long axis unknown without USD inspection).
SHELF_DECK_SIZE = (0.70, 0.70, 0.02)     # (width_x, depth_y, thickness_z) in meters
# Deck colour — red to match the rack frame. Tune to the exact rack USD red if needed.
# WARNING: visual materials on all 54 decks add material nodes; on Blackwell RTX 5050 this can
# re-trigger the camera SDP crash (bugs_errors/2026-05-22_sdp-camera-crash-blackwell.md, "108
# material nodes"). If the camera crashes, revert this or colour only the level-0 decks.
RACK_RED = (0.55, 0.06, 0.06)

RACK_SHELF_Z = RACK_SHELF_LEVELS[-1]     # top shelf z (kept for explore_scene hint)

# 18 target boxes: one per rack, on the FLOOR in front of the rack (within Franka reach).
# Category cycles fragile/regular/heavy by rack index -> 6 of each across 18 racks.
# Boxes are graspable rigid bodies; the commanded box is selected at runtime by goal_id.
# Sourced from layout_grid (pure-Python) so tests can verify specs without Isaac Sim.
TARGET_BOX_SPECS: list[tuple[str, float, float, tuple[float, float, float]]] = _TARGET_BOX_SPECS_LG

# Back-compat alias: env code iterates ITEM_SPECS.
ITEM_SPECS = TARGET_BOX_SPECS

ZONE_SIZE  = (3.0, 3.0, 0.02)
ZONE_SPECS = [
    ("zone_A", (1.0, 0.9, 0.0), (-6.0, -12.0, 0.01)),  # orange — fragile
    ("zone_B", (0.0, 0.9, 0.9), ( 0.0, -12.0, 0.01)),  # cyan   — regular
    ("zone_C", (0.7, 0.0, 0.9), ( 6.0, -12.0, 0.01)),  # purple — heavy
]

# Static props: (name, usd_path, (x, y, z))
# forklift + palletasm removed — unnecessary USD load, SDP pressure on Blackwell.
PROP_SPECS = [
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


def _shelf_deck_cfg(rack_idx: int, level_idx: int, rack_pos: tuple, shelf_z: float) -> AssetBaseCfg:
    """Static thin cuboid shelf deck — solid collision surface for boxes at shelf_z.

    Top surface at shelf_z; platform center offset down by half-thickness.
    3 decks per rack (RACK_SHELF_LEVELS) × 18 racks = 54 platforms total.
    """
    # Platform center z = shelf surface z - half thickness
    pos = (rack_pos[0], rack_pos[1], shelf_z - SHELF_DECK_SIZE[2] / 2.0)
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/shelf_{rack_idx}_{level_idx}",
        spawn=sim_utils.CuboidCfg(
            size=SHELF_DECK_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # Red deck to match the rack frame (user request 2026-06-17). NOTE: adds a material node
            # per deck (54 total) — this is the SDP BindMaterialCommand that previously crashed the
            # Blackwell RTX 5050 camera. See bugs_errors/2026-05-22_sdp-camera-crash-blackwell.md.
            # If the camera crashes on run, revert this line.
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=RACK_RED),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _item_cfg(name: str, size: float, mass: float, pos: tuple) -> RigidObjectCfg:
    """Rigid primitive cube — physics proxy for cardboard box.

    Uses CuboidCfg instead of UsdFileCfg because NVIDIA DT cardboard box USDs
    are visual-only assets (no RigidBodyAPI schema on root prim). RigidObjectCfg
    with UsdFileCfg raises 'Failed to find a rigid body' at sim init.

    Size (0.21/0.32/0.52m) encodes category for CLIP/YOLO.
    Color differentiates category visually at 64x64 resolution.
    BOX_USD paths are kept as constants for future use if assets gain RigidBodyAPI.

    Per-box visual_material (54 PreviewSurfaceCfg nodes): verified safe on driver 580.88
    — test_env.py ALL PASS 2026-06-03 with camera ON. Was previously suspected as SDP crash
    trigger on 591.x/595.x; keep driver pinned at 580.88 (see bugs_errors/sdp-camera-crash).
    """
    # Brown shades per category — light=fragile, medium=regular, dark=heavy.
    _COLORS = {0.21: (0.85, 0.70, 0.45), 0.32: (0.70, 0.52, 0.28), 0.52: (0.50, 0.37, 0.18)}
    color = _COLORS.get(size, (0.75, 0.60, 0.35))
    # Spawn 5cm above target shelf surface — falls onto shelf deck.
    raised_pos = (pos[0], pos[1], pos[2] + 0.05)
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(size, size, size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=raised_pos),
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

    robot: ArticulationCfg = RIDGEBACK_FRANKA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    camera: TiledCameraCfg = TiledCameraCfg(
        # Parent under base_link (the MOVING chassis), NOT /Robot (the welded `world` root, which
        # stays fixed at origin — IsaacLab #1268). Mounting on /Robot froze the camera at spawn so
        # every frame was identical while the robot drove. base_link prim verified at :445.
        prim_path="{ENV_REGEX_NS}/Robot/base_link/onboard_cam",
        update_period=0.1,
        height=64,
        width=64,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,          # 18mm → HFOV ~60° (was 24mm → 47°); matches RealSense D435
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 50.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.35, 0.0, 0.55),      # front-top of Ridgeback base, forward-facing
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

    # Contact sensor on the Ridgeback base — required for collision penalty (Phase 3).
    # base_link VERIFIED via smoke_test.py 2026-06-03 (robot.body_names includes "base_link").
    # NO filter_prim_paths_expr: collision_penalty reads net_forces_w_history (net force on the
    # chassis), not the per-object force_matrix. A filter expr must resolve to exactly 1 prim per
    # env; "Rack_.*"/"wall_.*" matched 18/4 → PhysX "expected 1, found 18" → corrupt GPU contact
    # view → CUDA illegal memory access on reset(). Net-force sensor needs no filter.
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.0,
        history_length=3,
    )

    def __post_init__(self) -> None:
        """Register racks + shelf decks + boxes + props as scene attributes.

        - 18 racks (static USD)
        - 54 shelf decks (18 racks × 3 levels, invisible CuboidCfg — solid collision surface)
        - 18 boxes (RigidObjectCfg — gravity, one per rack on the floor in front, within Franka reach)
        - 11 props (static USD)
        """
        for i, rack_pos in enumerate(RACK_POSITIONS):
            setattr(self, f"rack_{i}", _rack_cfg(i, rack_pos))
            for j, shelf_z in enumerate(RACK_SHELF_LEVELS):
                setattr(self, f"shelf_{i}_{j}", _shelf_deck_cfg(i, j, rack_pos, shelf_z))
        for name, size, mass, pos in ITEM_SPECS:
            setattr(self, name, _item_cfg(name, size, mass, pos))
        for name, usd_path, pos in PROP_SPECS:
            setattr(self, name, _prop_cfg(name, usd_path, pos))
