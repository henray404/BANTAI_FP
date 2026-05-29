# explore_rack.py
# Entry script — AppLauncher lives here.
# Loads Rack_A01 USD standalone, prints bounding box to determine RACK_SHELF_Z.
#
# Workflow:
#   1. Run this script (windowed). Look at rack in viewport.
#   2. Note the Z-output printed below — that is the estimated top-shelf height.
#   3. Update RACK_SHELF_Z in env/warehouse_scene.py.
#   4. Run explore_scene.py to verify boxes sit correctly on shelves.
#
# Usage:
#   conda activate isaaclab
#   python asset_sandbox/scripts/explore_rack.py

"""Rack_A01 standalone viewer: prints bounding box Z for RACK_SHELF_Z tuning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# ── CLI + AppLauncher ─────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Rack_A01 USD viewer")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports (after AppLauncher) ───────────────────────────────────────
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from env.warehouse_scene import RACK_SHELF_Z, RACK_USD


def _print_rack_bounds(prim_path: str) -> None:
    """Print bounding box Z range and estimated top-shelf height."""
    try:
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"[WARN] Prim not found at {prim_path}")
            return
        bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(prim)
        box = bbox.GetBox()
        mn, mx = box.GetMin(), box.GetMax()
        meters_per_unit = 0.01  # Rack_A01 authored in cm; BBoxCache returns USD units
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
        if abs(estimated - RACK_SHELF_Z) > 0.1:
            print(f"[ACTION] Update RACK_SHELF_Z to {estimated} in env/warehouse_scene.py")
        else:
            print("[OK] RACK_SHELF_Z looks correct.")
    except Exception as exc:
        print(f"[WARN] Could not compute bounds: {exc}")


def main() -> None:
    """Load rack, print bounds, keep window open for visual inspection."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, render_interval=1)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(3.0, -2.5, 3.5), target=(0.0, 0.0, 1.5))

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

    rack_cfg = sim_utils.UsdFileCfg(
        usd_path=RACK_USD,
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    rack_cfg.func("/World/Rack", rack_cfg)

    sim.reset()
    print(f"[INFO] Rack loaded: {RACK_USD}")
    _print_rack_bounds("/World/Rack")
    print("[INFO] Inspect rack in viewport. Close window to exit.")

    step = 0
    while simulation_app.is_running():
        sim.step()
        step += 1
        if step % 500 == 0:
            print(f"[step {step}] Running — close window when done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
