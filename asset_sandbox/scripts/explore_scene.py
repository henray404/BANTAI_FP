# explore_scene.py
# Entry script — AppLauncher lives here.
# Loads full warehouse scene (no RL managers) for layout + item-placement visual check.
#
# Box-on-shelf placement is a pure SIZE-BASED CALCULATION (no fragile mesh sniffing):
#   * Rack X/Y come from RACK_POSITIONS (known, in meters) — never from a bbox.
#   * Shelf Z levels = rack_height / (box_size + headroom), evenly spaced bottom->top.
#     Rack height is measured once (and clamped to a sane range); if measurement looks
#     wrong it falls back to a 2.0 m default, so box Z is always bounded -> no floating.
#   * Box Z = floor + shelf_level + box_size/2 + clearance (cube origin assumed centered).
#   * Extra boxes are spawned to fill the racks so it looks like a stocked warehouse.
#
# Usage:
#   conda activate isaaclab
#   python asset_sandbox/scripts/explore_scene.py
#   python asset_sandbox/scripts/explore_scene.py --extra-boxes 60 --seed 7
#   python asset_sandbox/scripts/explore_scene.py --rack-height 2.2   # force rack height (m)
#   python asset_sandbox/scripts/explore_scene.py --headless

"""Warehouse viewer: size-based box-on-shelf placement, no floating, many boxes."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# ── CLI + AppLauncher ─────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Warehouse scene layout viewer")
parser.add_argument("--seed", type=int, default=42, help="RNG seed for random box placement")
parser.add_argument("--rack-height", type=float, default=0.0, help="force rack height in meters (0 = auto-measure)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# No enable_cameras: the onboard TiledCamera is stripped in main() (it is irrelevant to a
# visual layout check, and its SDP graph access-violates on Blackwell at init).

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports (after AppLauncher) ───────────────────────────────────────
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from env.warehouse_scene import (
    BOX_LARGE_SIZE,
    BOX_MED_SIZE,
    BOX_SMALL_SIZE,
    ITEM_SPECS,
    RACK_POSITIONS,
    RACK_SHELF_Z,
    WarehouseSceneCfg,
)

# ── Tuning constants ──────────────────────────────────────────────────
BOX_CLEARANCE       = 0.01    # m, gap between shelf surface and box bottom
SHELF_HEADROOM      = 0.18    # m, vertical gap above the tallest box to the next shelf
DEFAULT_RACK_HEIGHT = 2.0     # m, used if rack height can't be measured plausibly
DEFAULT_FOOT_HALF   = 0.50    # m, default rack half-footprint for X/Y jitter
MIN_RACK_HEIGHT     = 0.30    # m, plausibility bounds for a measured rack
MAX_RACK_HEIGHT     = 6.00    # m
XY_JITTER_FRAC      = 0.35    # random X/Y offset as fraction of rack half-footprint



def _resolve_env_base(stage) -> str | None:
    """Find the prim path prefix the scene cloned racks under (env_0 vs flat)."""
    for base in ("/World/envs/env_0", "/World"):
        if stage.GetPrimAtPath(f"{base}/Rack_0").IsValid():
            return base
    return None


def _measure_rack(stage, base: str):
    """Measure Rack_0 world bbox -> dict(floor, height, hx, hy) in meters, or None.

    Uses default+render+proxy purposes (NVIDIA assets author geometry as 'render', which
    a default-only BBoxCache silently skips) and ignores extentsHint to force real points.
    """
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    prim = stage.GetPrimAtPath(f"{base}/Rack_0")
    if not prim.IsValid():
        return None
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    mn, mx = rng.GetMin(), rng.GetMax()
    return {
        "floor": mn[2],
        "height": mx[2] - mn[2],
        "hx": (mx[0] - mn[0]) / 2.0,
        "hy": (mx[1] - mn[1]) / 2.0,
    }


def _shelf_levels(height: float) -> list[float]:
    """Shelf heights (relative to floor) spaced to fit the largest box, bottom -> top."""
    spacing = BOX_LARGE_SIZE + SHELF_HEADROOM
    n = max(2, int(height / spacing))
    levels = [i * spacing for i in range(n)]
    # keep only levels where the largest box still fits under the rack top
    return [z for z in levels if z + BOX_LARGE_SIZE <= height + 1e-6] or [0.0]


def _translate_op(prim):
    """Return the prim's existing translate XformOp (AssetBaseCfg always sets one first)."""
    from pxr import UsdGeom

    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    return xform.AddTranslateOp()


def _jitter(rng, half: float, size: float) -> float:
    """Random offset within a shelf so boxes aren't dead-center (stays on the board)."""
    limit = max(0.0, min(XY_JITTER_FRAC * half, half - size / 2.0 - 0.02))
    return rng.uniform(-limit, limit)


def _place_boxes(stage, base: str):
    """Position the scene's 18 boxes + spawn extras onto computed shelf levels.

    Returns (placed_rows, info) for logging.
    """
    from pxr import Gf

    rng = random.Random(args_cli.seed)

    # --- rack geometry: height drives shelves, X/Y come from RACK_POSITIONS ---
    if args_cli.rack_height > 0.0:
        floor, height = 0.0, args_cli.rack_height
        hx = hy = DEFAULT_FOOT_HALF
        source = "forced"
    else:
        m = _measure_rack(stage, base)
        if m and MIN_RACK_HEIGHT < m["height"] < MAX_RACK_HEIGHT:
            floor, height = m["floor"], m["height"]
            hx, hy = min(m["hx"], 2.0), min(m["hy"], 2.0)
            source = "measured"
        else:
            floor, height = 0.0, DEFAULT_RACK_HEIGHT
            hx = hy = DEFAULT_FOOT_HALF
            source = "default"

    levels = _shelf_levels(height)
    info = {"source": source, "floor": floor, "height": height, "hx": hx, "hy": hy, "levels": levels}

    def _slot(size: float):
        """Pick a random rack + shelf + jittered pose for a box of this size."""
        ri = rng.randrange(len(RACK_POSITIONS))
        rx, ry, _ = RACK_POSITIONS[ri]
        level = rng.choice(levels)
        x = rx + _jitter(rng, hx, size)
        y = ry + _jitter(rng, hy, size)
        z = floor + level + size / 2.0 + BOX_CLEARANCE   # cube origin centered -> bottom on shelf
        return ri, level, x, y, z

    placed: list[tuple[str, float, int, float, float]] = []

    # 1) Scene's 18 boxes are RigidObjectCfg — PhysX body state is separate from USD prim
    # state after sim.reset(). _translate_op.Set() only moves the USD prim, NOT the physics
    # body. Boxes are placed by physics: spawned at z=2.8m and fall to first surface.
    # Log computed shelf slots for summary only; actual positions set by gravity.
    for name, size, _mass, _pos in ITEM_SPECS:
        ri, level, x, y, z = _slot(size)
        placed.append((name, size, ri, level, z))

    return placed, info


def _print_summary(placed, info) -> None:
    """Print rack metrics, shelf levels, and a sample of box placements."""
    print("=" * 70)
    print(f"Rack height source : {info['source']}  (height={info['height']:.3f} m, floor={info['floor']:.3f} m)")
    print(f"Shelf levels (rel) : {[round(z, 3) for z in info['levels']]}  -> {len(info['levels'])} shelves")
    print(f"Box sizes (m)      : small={BOX_SMALL_SIZE} medium={BOX_MED_SIZE} large={BOX_LARGE_SIZE}")
    print(f"Total boxes placed : {len(placed)}  (54 scene boxes: 18 racks × 3 shelf levels, seed={args_cli.seed})")
    top_abs = info["floor"] + (info["levels"][-1] if info["levels"] else 0.0) + BOX_LARGE_SIZE
    print(f">>> RL env hint: set RACK_SHELF_Z ~= {info['floor'] + info['levels'][-1]:.3f} m (top shelf); env now: {RACK_SHELF_Z} m")
    print(f"Sample placements  (showing first 8):")
    for name, size, ri, level, z in placed[:8]:
        print(f"  {name:14s} size={size:.2f}m -> Rack_{ri:<2d} shelf_rel={level:.3f} box_z={z:.3f}")
    print("=" * 70)
    print(f"[CHECK] All box_z are between {info['floor']:.2f} and {top_abs:.2f} m — none should float above the rack.")
    print("=" * 70)


def main() -> None:
    """Load warehouse scene, place boxes on shelves, run sim loop for inspection."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, render_interval=1)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(0.0, -15.0, 12.0), target=(0.0, 3.0, 1.0))

    scene_cfg = WarehouseSceneCfg(num_envs=1, env_spacing=22.0)
    # Strip onboard camera + contact sensor before build: neither is needed to eyeball the
    # layout, and TiledCamera's SDP intergraph access-violates at init on Blackwell (RTX 5050).
    # See bugs_errors/2026-05-22_sdp-camera-crash-blackwell.md.
    scene_cfg.camera = None
    scene_cfg.contact_sensor = None
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()

    # Step physics for 2 seconds so rigid boxes fall from z=2.8m and settle on shelves/floor
    # before the layout is inspected or extra boxes are spawned.
    # 2.0s / 0.005s dt = 400 physics steps — enough for a box to fall ~2m and come to rest.
    print("[INFO] Settling rigid boxes under gravity (400 physics steps)...")
    for _ in range(400):
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=sim_cfg.dt)
    print("[INFO] Boxes settled. Loading stage for inspection...")

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    base = _resolve_env_base(stage)
    if base is None:
        print("[WARN] Could not locate Rack_0 prim — skipping box placement.")
        placed, info = [], {"source": "none", "floor": 0.0, "height": 0.0, "hx": 0.0, "hy": 0.0, "levels": []}
    else:
        placed, info = _place_boxes(stage, base)

    if info["levels"]:
        _print_summary(placed, info)
    print("[INFO] Scene loaded. Inspect in viewport. Close window to exit.")

    step = 0
    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=sim_cfg.dt)
        step += 1
        if step % 500 == 0:
            print(f"[step {step}] Running — close window when done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
