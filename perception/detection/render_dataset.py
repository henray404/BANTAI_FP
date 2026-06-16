# perception/detection/render_dataset.py
# Person 3 — render a YOLOv8 detection dataset from the warehouse env camera.
#
# ENTRY SCRIPT: owns AppLauncher (the env/ modules must not — see
# bugs_errors/2026-05-15_double-applaunch-crash.md). Steps the env with a random
# policy, saves each 64×64 frame + YOLO-format labels (auto-derived ground truth:
# every box's 3D pose+size is projected into the camera to a 2D bbox).
#
# UNVERIFIED on this hardware: the full env (needs `pixels`) has not run end-to-end
# on the RTX 5050 (Blackwell camera SDP blocker — docs/project/project_overview.md). Run on
# a working sim. Also VERIFY the camera convention/intrinsics math below on the first
# rendered frame (overlay the boxes) before trusting labels.
#
# Usage:
#   python perception/detection/render_dataset.py --frames 2000 --out perception/detection/dataset
#   python perception/detection/render_dataset.py --frames 2000 --headless

"""Render + auto-label a YOLOv8 detection dataset from the env camera."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render YOLOv8 dataset from warehouse camera")
parser.add_argument("--frames", type=int, default=2000, help="Frames (env steps) to capture")
parser.add_argument("--out", type=str, default="perception/detection/dataset", help="Output dir")
parser.add_argument("--val_frac", type=float, default=0.2, help="Validation split fraction")
parser.add_argument("--min_box_px", type=int, default=3, help="Drop bboxes smaller than this (px)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Project imports (after AppLauncher) ───────────────────────────────
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from env.warehouse_env import WarehouseEnvCfg, WarehouseRLEnv  # noqa: E402
from env.warehouse_scene import ITEM_SPECS  # noqa: E402

CATEGORY_ID = {"fragile": 0, "regular": 1, "heavy": 2}


def _box_corners(center: np.ndarray, size: float) -> np.ndarray:
    """8 cube corners (8,3) world-frame for an axis-aligned box of edge `size`."""
    h = size / 2.0
    o = np.array([[sx, sy, sz] for sx in (-h, h) for sy in (-h, h) for sz in (-h, h)])
    return center[None, :] + o


def _project(points_w: np.ndarray, cam_pos: np.ndarray, cam_rot: np.ndarray,
             K: np.ndarray, hw: int) -> np.ndarray | None:
    """Project world points (N,3) → pixel (N,2). ROS cam frame: x right, y down, z fwd.

    cam_rot = world→cam rotation (3,3). Returns None if all points are behind cam.
    VERIFY this convention on a real frame: if bboxes are mirrored/offset, the
    quaternion handedness or axis order is wrong (TiledCamera convention='ros').
    """
    pc = (points_w - cam_pos[None, :]) @ cam_rot.T  # world → cam
    z = pc[:, 2]
    if np.all(z <= 1e-3):
        return None
    z = np.clip(z, 1e-3, None)
    u = K[0, 0] * pc[:, 0] / z + K[0, 2]
    v = K[1, 1] * pc[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=-1)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) → 3×3 rotation matrix (cam→world)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def main() -> None:
    """Build env (num_envs=1), step randomly, save frames + YOLO labels."""
    out = Path(args_cli.out)
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)

    cfg = WarehouseEnvCfg()
    cfg.scene.num_envs = 1
    env = WarehouseRLEnv(cfg=cfg)
    cam = env.scene["camera"]
    hw = 64

    env.reset()
    saved = 0
    step = 0
    while saved < args_cli.frames and simulation_app.is_running():
        action = torch.from_numpy(
            np.random.uniform(-1, 1, size=(1, 3)).astype(np.float32)
        ).to(env.device)
        env.step(action)
        step += 1

        rgb = cam.data.output["rgb"][0, ..., :3]
        if rgb.dtype != torch.uint8:
            rgb = (rgb.clamp(0, 1) * 255).to(torch.uint8)
        img = rgb.cpu().numpy()

        K = cam.data.intrinsic_matrices[0].cpu().numpy()
        cam_pos = cam.data.pos_w[0].cpu().numpy()
        cam_rot_c2w = _quat_to_rot(cam.data.quat_w_ros[0].cpu().numpy())
        cam_rot_w2c = cam_rot_c2w.T

        lines: list[str] = []
        for name, size, _mass, _pos in ITEM_SPECS:
            cat = name.split("_")[0]
            center = (env.scene[name].data.root_pos_w[0] - env.scene.env_origins[0]).cpu().numpy()
            center_w = env.scene[name].data.root_pos_w[0].cpu().numpy()
            uv = _project(_box_corners(center_w, size), cam_pos, cam_rot_w2c, K, hw)
            if uv is None:
                continue
            x0, y0 = uv[:, 0].min(), uv[:, 1].min()
            x1, y1 = uv[:, 0].max(), uv[:, 1].max()
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(hw, x1), min(hw, y1)
            if x1 - x0 < args_cli.min_box_px or y1 - y0 < args_cli.min_box_px:
                continue
            cx, cy = (x0 + x1) / 2 / hw, (y0 + y1) / 2 / hw
            bw, bh = (x1 - x0) / hw, (y1 - y0) / hw
            lines.append(f"{CATEGORY_ID[cat]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            continue  # skip empty frames — no labeled box in view

        split = "val" if np.random.rand() < args_cli.val_frac else "train"
        stem = f"frame_{saved:06d}"
        _save_png(img, out / "images" / split / f"{stem}.png")
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        saved += 1
        if saved % 100 == 0:
            print(f"[render] saved {saved}/{args_cli.frames} (env step {step})")

    print(f"[render] done. {saved} labeled frames → {out}")
    env.close()


def _save_png(img: np.ndarray, path: Path) -> None:
    """Save an HWC uint8 array as PNG (imageio, fallback PIL)."""
    try:
        import imageio.v2 as imageio

        imageio.imwrite(path, img)
    except ImportError:
        from PIL import Image

        Image.fromarray(img).save(path)


if __name__ == "__main__":
    main()
    simulation_app.close()
