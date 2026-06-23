# s4 — Transfer Test (best model on a modified rack)

**Owner:** P5 · **Created:** 2026-06-23

The four study scenarios:

| scenario | what | where |
|----------|------|-------|
| s1 | train model-free baseline **PPO** | `c2_ppo` (`experiments/configs.py`) |
| s2 | train **DreamerV3** only | `c3_dreamer_vanilla` |
| s3 | train **DreamerV3 + CA-SLOPE** | `c4_dreamer_caslope` |
| **s4** | **best of s1–s3, evaluated on a modified scene** | this doc |

s1–s3 are the existing training configs (Visual HER configs kept in code, untouched — Henry owns
their removal). s4 adds **no training**: it loads the best checkpoint and evaluates generalization
on a warehouse the model never saw.

## What changes in the s4 scene

Defined in [`configs/env_config_s4.yaml`](../../configs/env_config_s4.yaml). Only the `scene:` block
is read from it (`env/warehouse_scene._scene_block` honours `$WAREHOUSE_ENV_CONFIG`); robot, reward,
termination, and episode length stay identical to training. Two changes:

1. **Physically smaller rack (whole rack + boxes + decks scale together)** — `Rack_L01` USD scale
   `0.01 → 0.007` (30% smaller). Shelf surfaces follow it: bottom `0.72 → 0.51 m`, **mid
   `1.33 → 0.93 m`**, top `1.93 → 1.35 m`. The boxes and shelf decks scale by the same factor
   (box `0.21/0.32/0.52 → 0.147/0.224/0.364 m`, deck `0.70 → 0.49 m`) so a box still fits on the
   smaller rack instead of clipping through the frame. (Category is commanded by `goal_id`, not box
   size — CLIP/YOLO removed — so shrinking the physical box is safe.)
2. **Extra box on the 2nd shelf level** — each rack keeps its bottom-shelf box AND gets an
   additional box on shelf level 1 (mid). The model trained only on bottom-shelf boxes must also
   grasp from a new height.
3. **Randomized placement (per-episode)** — which box (size) sits on which (rack, shelf) slot is
   shuffled on every reset, so any shelf can hold any size — not the fixed small/medium/big order.
   Box size is baked at spawn, so the box itself moves to a random slot (size travels with it);
   target selection (by category) and `box_pos` (read from the live pose) stay correct.

Pure layout math lives in `env/layout_grid.py` (`scale_shelf_levels`, `rest_box_specs`,
`extra_shelf_box_specs`, `shuffled_box_layout`), unit-tested in `tests/test_s4_layout.py` (no Isaac).
The per-episode reshuffle is `env/warehouse_env._randomize_box_poses` (gated by
`warehouse_scene.S4_RANDOMIZE_PLACEMENT`). The default scene (scale 0.01, placement off) is
byte-identical to before — all s4 behaviour is fully gated behind the yaml.

## How to run

```bash
conda activate isaaclab

# auto-pick the best run across s1-s3 results, eval on the s4 scene:
python scripts/eval_s4.py --auto --results training/results/ablation --episodes 20

# or eval one explicit run:
python scripts/eval_s4.py --run training/results/ablation/c2_ppo_seed0 --algo ppo
```

`scripts/eval_s4.py` sets `WAREHOUSE_ENV_CONFIG=configs/env_config_s4.yaml` before importing the env,
loads the SB3 (PPO/SAC) best model, runs deterministic eval, and writes
`training/results/s4/s4_metrics.json` + the eval CSV.

**DreamerV3 best model:** the actor loader is not wired into `eval_s4.py` (NM512 trainer-specific).
Keep `WAREHOUSE_ENV_CONFIG=configs/env_config_s4.yaml` set and run the DreamerV3 eval path
(`experiments/nm512_eval`) against this env with the checkpoint — the scene override is identical;
only the actor load differs.

## Sim-verify (Blackwell-blocked — run on Linux/A100)

- Confirm the arm reaches the mid shelf at `z ≈ 1.09 m`. If not, lower `racks.scale` (e.g. `0.006`)
  or set `second_shelf_box.shelf_level: 0` in the s4 yaml.
- Confirm the shrunk rack's baked collider still blocks the base (smaller footprint).
