# perception/detection/train.py
# Person 3 — train YOLOv8 on the rendered warehouse box dataset.
#
# No Isaac/AppLauncher needed — pure ultralytics. Run AFTER render_dataset.py.
#   python perception/detection/train.py
#   python perception/detection/train.py --model yolov8s.pt --epochs 200
#
# Requires `pip install ultralytics` (requirements-ml.txt).

"""YOLOv8 training entry for the size-coded box detector."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Load config.yaml, override via CLI, train YOLOv8."""
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    p = argparse.ArgumentParser(description="Train YOLOv8 box detector")
    p.add_argument("--model", default=cfg["model"])
    p.add_argument("--imgsz", type=int, default=cfg["imgsz"])
    p.add_argument("--epochs", type=int, default=cfg["epochs"])
    p.add_argument("--batch", type=int, default=cfg["batch"])
    p.add_argument("--device", default=str(cfg["device"]))
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("ultralytics not installed — pip install -r requirements-ml.txt") from e

    model = YOLO(args.model)
    model.train(
        data=str(HERE / cfg["data"]),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        patience=cfg["patience"],
        project=str(HERE / cfg["project"]),
        name=cfg["name"],
    )
    metrics = model.val()
    print(f"[yolo] mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
