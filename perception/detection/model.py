# perception/detection/model.py
# Person 3 — YOLOv8 inference wrapper used by SLOPE at training time.
"""Inference wrapper around a trained YOLOv8 box detector."""

from __future__ import annotations

import numpy as np

CATEGORIES = ("fragile", "regular", "heavy")


class BoxDetector:
    """Wrap a trained YOLOv8 weight for per-frame box detection.

    detect(image) → list of (category, confidence, (cx, cy, w, h)) in normalized
    image coords. Consumed by slope.py to build the category-conditioned potential.
    """

    def __init__(self, weights: str, device: str = "cpu", conf: float = 0.25):
        """Load YOLO weights (lazy ultralytics import)."""
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device
        self.conf = conf

    def detect(self, image: np.ndarray):
        """Run detection on an HWC uint8 (or float[0,1]) image."""
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        res = self.model.predict(image, device=self.device, conf=self.conf, verbose=False)[0]
        out = []
        for b in res.boxes:
            cls = int(b.cls.item())
            conf = float(b.conf.item())
            xywhn = b.xywhn[0].cpu().numpy()  # normalized cx,cy,w,h
            out.append((CATEGORIES[cls], conf, tuple(xywhn.tolist())))
        return out
