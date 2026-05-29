# explore_scene.py
# Entry script — AppLauncher lives here.
# Loads full warehouse scene (no RL managers) for layout + item-placement visual check.
#
# Run AFTER explore_rack.py confirms RACK_SHELF_Z is correct.
#
# Usage:
#   conda activate isaaclab
#   python asset_sandbox/scripts/explore_scene.py
#   python asset_sandbox/scripts/explore_scene.py --headless  # render without window

"""Full warehouse layout viewer (no RL env): racks, items, zones, walls, robot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# ── CLI + AppLauncher ─────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Warehouse scene layout viewer")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # TiledCamera requires this flag

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


def _print_layout_summary() -> None:
    """Print current layout constants for quick sanity check."""
    print("=" * 50)
    print(f"RACK_SHELF_Z : {RACK_SHELF_Z} m")
    print(f"BOX sizes    : fragile={BOX_SMALL_SIZE} m | regular={BOX_MED_SIZE} m | heavy={BOX_LARGE_SIZE} m")
    print(f"Rack positions ({len(RACK_POSITIONS)} racks):")
    for i, pos in enumerate(RACK_POSITIONS):
        print(f"  rack_{i}: {pos}")
    print(f"Item positions ({len(ITEM_SPECS)} items):")
    for name, size, pos in ITEM_SPECS:
        print(f"  {name}: size={size:.2f}m  z={pos[2]:.3f}m")
    print("=" * 50)
    print("[CHECK] If boxes float above shelf -> decrease RACK_SHELF_Z")
    print("[CHECK] If boxes clip into rack    -> increase RACK_SHELF_Z")
    print("[CHECK] Update env/warehouse_scene.py and re-run this script.")
    print("=" * 50)


def main() -> None:
    """Load warehouse scene, run simulation loop for visual inspection."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, render_interval=1)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(0.0, -15.0, 12.0), target=(0.0, 3.0, 1.0))

    scene_cfg = WarehouseSceneCfg(num_envs=1, env_spacing=22.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()

    _print_layout_summary()
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
