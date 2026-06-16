# Environment Specification — Warehouse Scene & RL Env

> Canonical numeric source: `configs/env_config.yaml`. Scene built in `env/warehouse_scene.py`;
> RL env (obs/action/reward/termination + Gym wrapper) in `env/warehouse_env.py`.
> Broader teammate design doc: `docs/environment.md`. This file is the focused parameter spec.

## Simulation / episode
| Param | Value |
|---|---|
| Simulator | Isaac Lab 5.1, Windows 11, RTX 5050 8GB, Python 3.11, CUDA 12.8 |
| Physics | 200 Hz (`dt=0.005`) |
| Control | 10 Hz (`decimation=20`) |
| Episode length | 100 s × 10 Hz = **1000 steps** |
| Parallel envs | **1** (Ridgeback-Franka + 18 rigid boxes + arm IK on 8 GB VRAM) |
| `env_spacing` | 32 m (exceeds 30 m room) |

## Room
- Rectangular **20 × 30 m** — `x ∈ [-10, 10]`, `y ∈ [-15, 15]`.
- Walls: height 6 m, thickness 0.3 m.
- Ground friction: static 0.8, dynamic 0.6, restitution 0.0.
- Out-of-bounds (termination): `|x| > 9.5` or `|y| > 14.5`.

## Layout — 9 island grid (3×3)
- Island centers: `cols_x = [-6, 0, 6]`, `rows_y = [8, 1, -5]`.
- 2 racks per island → **18 racks total** (`island_rack_dx = 1.5`).
- Rack USD: `Rack_L01_PR_NVD_01.usd`, scale 0.01 (cm→m).
- Shelf surface z levels (measured 2026-05-30): `[0.724, 1.325, 1.926]`; deck size `0.70×0.70×0.02`.

## Graspable boxes (targets)
| Field | Value |
|---|---|
| Count | **18** — one per rack |
| Type | `RigidObjectCfg + CuboidCfg` (gravity + mass; NOT USD — DT box USDs lack RigidBodyAPI) |
| Placement | floor in front of rack (−y 0.5 m), within Franka reach (~0.85 m) |
| Randomize | x,y jitter each episode (`_randomize_box_poses`) |
| Selection | target chosen by `goal_id` one-hot (category) — **NOT detected** (no CLIP/YOLO) |

| Category | Size | Mass | Color (brown shade) |
|---|---|---|---|
| fragile | 0.21 m | 2.0 kg | light |
| regular | 0.32 m | 6.0 kg | medium |
| heavy | 0.52 m | 12.0 kg | dark |

## Delivery zones (3, size 3×3 m, at y = −12)
| Zone | Color | Position | Category |
|---|---|---|---|
| zone_A | yellow `[1.0, 0.9, 0.0]` | `[-6, -12, 0.01]` | fragile |
| zone_B | cyan `[0.0, 0.9, 0.9]` | `[ 0, -12, 0.01]` | regular |
| zone_C | purple `[0.7, 0.0, 0.9]` | `[ 6, -12, 0.01]` | heavy |

> Zone index order = category order (`fragile, regular, heavy`) = `goal_id` one-hot index.

## Props
- 2 plastic pallets (`[-3,-7,0]`, `[4,-7,0]`), 4 traffic cones, 2 warning signs.
- Forklift + pallet-assembly **removed** 2026-06-01 (SDP memory pressure on Blackwell).

## Spawn
- Receiving area north: `x ∈ [-8, 8]`, `y ∈ [11, 14]`, yaw random.

## Observation space (v2 — DO NOT change without team discussion)
```python
obs = {
    "pixels":   (B, 3, 64, 64),  # onboard RGB, float [0,1]
    "position": (B, 3),          # robot base xyz, env-local
    "heading":  (B, 2),          # [cos(yaw), sin(yaw)]
    "goal":     (B, 3),          # delivery zone xyz (anneals to zeros in curriculum)
    "goal_id":  (B, 3),          # one-hot [fragile,regular,heavy] — selects box + zone
    "ee_pos":   (B, 3),          # end-effector xyz, base frame
    "gripper":  (B, 1),          # finger opening 0..1
    "holding":  (B, 1),          # 1.0 if target box grasped
    "box_pos":  (B, 3),          # target box xyz, env-local — UNANNEALED (grasp needs precision)
}
```

## Reward (staged pick-place, switches on `holding`)
| Term | Weight | Phase |
|---|---|---|
| `-0.01 * dist(ee, box)` | approach (dense) | A — not holding |
| `+5.0 * grasp_success` | one-shot, box gripped + lifted | A |
| `-0.01 * dist(box, zone)` | carry (dense) | B — holding |
| `+10.0 * delivery_success` | held box in commanded color zone | B |
| `-0.005 * time_penalty` | efficiency | always |
| `-5.0 * collision` | chassis contact > 5 N | always |
| `-2.0 * drop_penalty` | box dropped outside zone | always |

## Termination
- Delivery success: box within **1.5 m** of commanded zone center.
- Out of bounds: `|x| > 9.5` or `|y| > 14.5`.

## Known constraints
- TiledCameraCfg only (CameraCfg crashes on Blackwell RTX 5050).
- Pin NVIDIA driver 580.88.
- GUI teleop choppy = Isaac main-thread (CPU core-0) bound, not GPU; cosmetic for physics.
