# docs/diagrams/gen_pipeline.py
# Generate docs/diagrams/pipeline.excalidraw — full BANTAI_FP ML pipeline diagram.
# Pure stdlib. Run: python docs/diagrams/gen_pipeline.py
# Open the .excalidraw at https://excalidraw.com or the VSCode Excalidraw extension.
"""Build a valid .excalidraw scene of the warehouse world-model pipeline."""

from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(7)
NONCE = lambda: random.randint(1, 2**31)
TS = 1717800000000

W, H = 250, 100  # default node box size

# ── Color groups ──────────────────────────────────────────────────────
C = {
    "env":   ("#1971c2", "#a5d8ff"),
    "obs":   ("#495057", "#ced4da"),
    "perc":  ("#2f9e44", "#b2f2bb"),
    "agent": ("#6741d9", "#d0bfff"),
    "infra": ("#f08c00", "#ffec99"),
}

# ── Nodes: id -> (x, y, group, label) ─────────────────────────────────
NODES = {
    "sim":    (40,   60,  "env",   "Isaac Lab Sim\nWarehouseSceneCfg\nracks · 54 boxes · zones · TiledCamera"),
    "rl":     (360,  60,  "env",   "WarehouseRLEnv (MDP)\nreset · reward · terminations\nper-env goal_pos"),
    "gym":    (680,  60,  "env",   "WarehouseGymEnv\naction (2,) [lin, ang]\nobs dict"),
    "obs":    (1010, 60,  "obs",   "obs\npixels · position · goal\ngoal_emb · heading"),
    "clip":   (680,  -150, "perc", "CLIP encoder (P4)\nperception/language\nViT-B/32 frozen · instructions"),
    "render": (40,   330, "perc",  "render_dataset.py (P3)\nframes + labels\n(3D box -> camera proj)"),
    "ytrain": (360,  330, "perc",  "YOLOv8 (P3)\ntrain.py · model.py\nBoxDetector"),
    "slope":  (680,  330, "perc",  "Category-Aware SLOPE (P3)\nslope.py\n-> auxiliary reward"),
    "dreamer":(1360, -120, "agent","DreamerV3 (P2)\nobs_adapter -> warehouse_dreamer_env\nvendor RSSM + actor-critic"),
    "sac":    (1360, 60,  "agent", "SAC / PPO baseline (P5)\nenv_adapter -> SB3\nMultiInputPolicy"),
    "her":    (1010, 330, "infra", "Visual HER (P4)\nvisual_her.py\nrelabel by approached category"),
    "buffer": (1360, 330, "infra", "ReplayBuffer (P5)\nDict-obs ring\nadd / sample"),
    "trainer":(1710, 60,  "infra", "Trainer / experiment (P5)\nseed.py · configs/\nmulti-seed runner"),
    "wandb":  (1710, 330, "infra", "W&B logger.py (P5)\ncurves · checkpoints"),
}

# ── Edges: (src, dst, label) ──────────────────────────────────────────
EDGES = [
    ("sim", "rl", ""),
    ("rl", "gym", "step"),
    ("gym", "obs", "obs dict"),
    ("clip", "obs", "goal_emb"),
    ("obs", "render", "pixels"),
    ("render", "ytrain", "dataset"),
    ("ytrain", "slope", "detections"),
    ("slope", "rl", "aux reward"),
    ("obs", "dreamer", "obs"),
    ("obs", "sac", "obs"),
    ("dreamer", "gym", "action"),
    ("sac", "gym", "action"),
    ("gym", "buffer", "transitions"),
    ("buffer", "dreamer", "sample"),
    ("buffer", "sac", "sample"),
    ("her", "buffer", "relabel"),
    ("trainer", "dreamer", "drive"),
    ("trainer", "sac", "drive"),
    ("dreamer", "wandb", "log"),
    ("sac", "wandb", "log"),
]

elements: list[dict] = []
rect_id: dict[str, str] = {}
bound: dict[str, list] = {}  # element id -> boundElements list


def _rect(node_id: str, x, y, group, label):
    """Create a rounded rectangle + its centered bound text label."""
    rid = f"r_{node_id}"
    tid = f"t_{node_id}"
    rect_id[node_id] = rid
    stroke, bg = C[group]
    bound.setdefault(rid, []).append({"type": "text", "id": tid})
    elements.append({
        "id": rid, "type": "rectangle", "x": x, "y": y, "width": W, "height": H,
        "angle": 0, "strokeColor": stroke, "backgroundColor": bg, "fillStyle": "solid",
        "strokeWidth": 2, "strokeStyle": "solid", "roughness": 1, "opacity": 100,
        "groupIds": [], "frameId": None, "roundness": {"type": 3}, "seed": NONCE(),
        "version": 1, "versionNonce": NONCE(), "isDeleted": False,
        "boundElements": bound[rid], "updated": TS, "link": None, "locked": False,
    })
    lines = label.split("\n")
    elements.append({
        "id": tid, "type": "text", "x": x + 8, "y": y + H / 2 - len(lines) * 10,
        "width": W - 16, "height": len(lines) * 20, "angle": 0,
        "strokeColor": "#1e1e1e", "backgroundColor": "transparent", "fillStyle": "solid",
        "strokeWidth": 2, "strokeStyle": "solid", "roughness": 1, "opacity": 100,
        "groupIds": [], "frameId": None, "roundness": None, "seed": NONCE(),
        "version": 1, "versionNonce": NONCE(), "isDeleted": False, "boundElements": [],
        "updated": TS, "link": None, "locked": False, "text": label, "fontSize": 14,
        "fontFamily": 1, "textAlign": "center", "verticalAlign": "middle",
        "containerId": rect_id[node_id], "originalText": label, "lineHeight": 1.25,
    })


def _center(node_id):
    """Center point of a node box."""
    x, y, *_ = NODES[node_id]
    return x + W / 2, y + H / 2


def _edge_point(node_id, toward):
    """Point on node box boundary toward `toward` center."""
    cx, cy = _center(node_id)
    tx, ty = toward
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    sx = (W / 2) / abs(dx) if dx else float("inf")
    sy = (H / 2) / abs(dy) if dy else float("inf")
    s = min(sx, sy)
    return cx + dx * s, cy + dy * s


def _arrow(src, dst, label):
    """Create a bound arrow from src node to dst node with an optional label."""
    aid = f"a_{src}_{dst}"
    sx, sy = _edge_point(src, _center(dst))
    ex, ey = _edge_point(dst, _center(src))
    bind_s = rect_id[src]
    bind_e = rect_id[dst]
    bound.setdefault(bind_s, []).append({"type": "arrow", "id": aid})
    bound.setdefault(bind_e, []).append({"type": "arrow", "id": aid})
    a_bound = []
    el = {
        "id": aid, "type": "arrow", "x": sx, "y": sy,
        "width": abs(ex - sx), "height": abs(ey - sy), "angle": 0,
        "strokeColor": "#343a40", "backgroundColor": "transparent", "fillStyle": "solid",
        "strokeWidth": 2, "strokeStyle": "solid", "roughness": 1, "opacity": 100,
        "groupIds": [], "frameId": None, "roundness": {"type": 2}, "seed": NONCE(),
        "version": 1, "versionNonce": NONCE(), "isDeleted": False, "boundElements": a_bound,
        "updated": TS, "link": None, "locked": False, "points": [[0, 0], [ex - sx, ey - sy]],
        "lastCommittedPoint": None,
        "startBinding": {"elementId": bind_s, "focus": 0.1, "gap": 4},
        "endBinding": {"elementId": bind_e, "focus": 0.1, "gap": 4},
        "startArrowhead": None, "endArrowhead": "arrow",
    }
    elements.append(el)
    if label:
        lid = f"l_{src}_{dst}"
        a_bound.append({"type": "text", "id": lid})
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        elements.append({
            "id": lid, "type": "text", "x": mx - 30, "y": my - 10, "width": 60, "height": 20,
            "angle": 0, "strokeColor": "#343a40", "backgroundColor": "#ffffff",
            "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid", "roughness": 1,
            "opacity": 100, "groupIds": [], "frameId": None, "roundness": None, "seed": NONCE(),
            "version": 1, "versionNonce": NONCE(), "isDeleted": False, "boundElements": [],
            "updated": TS, "link": None, "locked": False, "text": label, "fontSize": 12,
            "fontFamily": 1, "textAlign": "center", "verticalAlign": "middle",
            "containerId": aid, "originalText": label, "lineHeight": 1.25,
        })


def _title():
    """Add a title text at the top."""
    elements.append({
        "id": "title", "type": "text", "x": 40, "y": -240, "width": 900, "height": 36,
        "angle": 0, "strokeColor": "#1e1e1e", "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid", "roughness": 1,
        "opacity": 100, "groupIds": [], "frameId": None, "roundness": None, "seed": NONCE(),
        "version": 1, "versionNonce": NONCE(), "isDeleted": False, "boundElements": [],
        "updated": TS, "link": None, "locked": False,
        "text": "BANTAI_FP — Visual Category-Aware World-Model Warehouse Robot (pipeline)",
        "fontSize": 24, "fontFamily": 1, "textAlign": "left", "verticalAlign": "top",
        "originalText": "BANTAI_FP — pipeline", "lineHeight": 1.25,
    })


def main():
    """Build all elements and write the .excalidraw file."""
    _title()
    for nid, (x, y, g, label) in NODES.items():
        _rect(nid, x, y, g, label)
    for src, dst, label in EDGES:
        _arrow(src, dst, label)

    scene = {
        "type": "excalidraw", "version": 2, "source": "bantai_fp/gen_pipeline.py",
        "elements": elements,
        "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
        "files": {},
    }
    out = Path(__file__).resolve().parent / "pipeline.excalidraw"
    out.write_text(json.dumps(scene, indent=2))
    print(f"wrote {out}  ({len(elements)} elements)")


if __name__ == "__main__":
    main()
