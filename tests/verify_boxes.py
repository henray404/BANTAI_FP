import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify boxes and prismatic frame")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import math
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import traceback

def main():
    try:
        from env.warehouse_env import WarehouseEnvCfg, WarehouseGymEnv
        cfg = WarehouseEnvCfg()
        cfg.scene.num_envs = args_cli.num_envs
        env = WarehouseGymEnv(cfg=cfg)
        obs, _ = env.reset()
        
        # Step to let boxes fall
        for _ in range(50):
            action = torch.zeros(args_cli.num_envs, 2, device=env.device)
            env.step(action)
        
        from env.warehouse_scene import ITEM_SPECS
        boxes_z = []
        floor_count = 0
        shelf_count = 0
        for name, _, _, _ in ITEM_SPECS:
            box = env._env.scene[name]
            z = box.data.root_pos_w[0, 2].item()
            if z < 0.3:
                floor_count += 1
            elif z > 0.3:
                shelf_count += 1
            boxes_z.append(z)
            
        with open("bugs_errors/_verify_results.txt", "w") as f:
            f.write(f"\n--- BOX VERIFICATION ---\n")
            f.write(f"Total boxes checked: {len(boxes_z)}\n")
            f.write(f"Boxes on floor (z < 0.3): {floor_count}\n")
            f.write(f"Boxes on shelf (z > 0.3): {shelf_count}\n")
            if floor_count > 0:
                f.write(f"FAILED: {floor_count} boxes fell to floor!\n")
            else:
                f.write(f"PASSED: All boxes are on shelves.\n")
                
            # Now check prismatic frame
            f.write(f"\n--- PRISMATIC FRAME VERIFICATION ---\n")
            robot = env._env.scene["robot"]
            
            # Check if base_link exists
            if "base_link" not in robot.body_names:
                f.write(f"ERROR: base_link not in robot.body_names! Found: {robot.body_names}\n")
                bidx = 0
            else:
                bidx = robot.body_names.index("base_link")
            
            p0 = robot.data.body_pos_w[0, bidx, :2].clone()
            
            # Drive forward
            for _ in range(100):
                action = torch.tensor([[1.0, 0.0]], device=env.device)
                env.step(action)
                
            p1 = robot.data.body_pos_w[0, bidx, :2].clone()
            dir1 = p1 - p0
            d1 = torch.linalg.norm(dir1).item()
            
            # Yaw
            for _ in range(100):
                action = torch.tensor([[0.0, 1.0]], device=env.device)
                env.step(action)
                
            # Drive forward again
            p2 = robot.data.body_pos_w[0, bidx, :2].clone()
            for _ in range(100):
                action = torch.tensor([[1.0, 0.0]], device=env.device)
                env.step(action)
                
            p3 = robot.data.body_pos_w[0, bidx, :2].clone()
            dir2 = p3 - p2
            d2 = torch.linalg.norm(dir2).item()
            
            f.write(f"Phase 1 disp: {d1:.3f}\n")
            f.write(f"Phase 3 disp: {d2:.3f}\n")
            
            if d1 < 0.02 or d2 < 0.02:
                f.write("INCONCLUSIVE\n")
            else:
                cos = torch.dot(dir1/d1, dir2/d2).item()
                f.write(f"Cos: {cos:.3f}\n")
                if cos > 0.7:
                    f.write("VERDICT: WORLD-FRAME\n")
                elif abs(cos) < 0.4:
                    f.write("VERDICT: BODY-FRAME\n")
                else:
                    f.write("VERDICT: AMBIGUOUS\n")
                    
            f.write("------------------------\n")
            
        print("Verification complete. Results in bugs_errors/_verify_results.txt")
        import os
        os._exit(0)
    except Exception as e:
        with open("bugs_errors/_verify_error.txt", "w") as f:
            f.write(traceback.format_exc())
        import os
        os._exit(1)

if __name__ == "__main__":
    main()
