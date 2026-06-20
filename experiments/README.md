# experiments/ — CA-SLOPE eval harness (P5)

> Penjelasan konsep CA-SLOPE + cara nyambungin ke training DreamerV3 (P2/P3): **`docs/ca_slope.md`**.
> File ini cuma soal harness eval headless-nya.


Headless evaluation of the warehouse pickup task. Records the robot's **per-step trace** and a
**per-episode summary** to CSV, and applies **CA-SLOPE** (Category-Aware SLOPE) shaping so we can
measure the RQ2 ablation: per-category reward landscape vs. one generic landscape.

Runs on a **Mac, no Isaac Lab, no torch** — pure numpy + stdlib. Isaac Lab can't run on macOS, and
the DreamerV3 policy is still being fixed, so this is a deliberately rough (`kasaran`) stand-in that
reproduces the obs/action/reward **contract**, not the physics. Swap in the real env + learned
policy later without touching the CSV/metrics path.

## Run it

```bash
python experiments/run_eval.py                 # CA-SLOPE (category-aware), 3 scenarios x 3 seeds
python experiments/run_eval.py --mode generic  # generic SLOPE (one gain) — RQ2 control
python experiments/run_eval.py --mode none     # no shaping (vanilla base reward)
python experiments/run_eval.py --ablation      # all three modes back-to-back (RQ2 sweep)
python experiments/run_eval.py --seeds 0 1 2 3 4 --out training/results/eval
```

Output (under `--out`, default `training/results/eval/`, git-ignored):

- `steps_<mode>.csv` — one row per robot step (rekam jejak): pose, ee/box/goal xyz, distances,
  action (6), `base_reward`, `slope_reward`, `total_reward`, cumulative returns, grasp/deliver/drop
  events, `done`, `success`.
- `summary_<mode>.csv` — one row per scenario × seed: success, steps, grasp/deliver step, returns,
  final box→zone distance.

## RQ2 — what to look at

`mode=none` `total_reward == base_reward`. `mode=generic` and `mode=category` leave `base_reward`
unchanged (potential-based shaping is policy-invariant by construction) but reshape the dense
landscape. In `category` mode the per-category gains (fragile 1.0 / regular 1.5 / heavy 2.0) spread
`return_total` by category; `generic` collapses them to one gain. That spread is the CA-SLOPE effect.

## Pieces

| File | Role |
|---|---|
| `../reward/ca_slope.py` | CA-SLOPE potential + shaping. Backend-agnostic (numpy here, torch in the Isaac reward). |
| `scenarios.py` | One scenario per category; coords mirror the real scene constants. |
| `toy_pickup_env.py` | Numpy kinematic stand-in reproducing the obs/action/reward contract. |
| `scripted_policy.py` | Greedy placeholder controller. Replace with the DreamerV3 actor when ready. |
| `eval_harness.py` | Runs scenarios, applies CA-SLOPE, streams the two CSVs. |
| `run_eval.py` | CLI. |

## Wiring to the real stack later

- Real env: `EvalHarness` only needs `reset(scenario, seed)` / `step(action)` and the obs keys
  (`ee_pos, box_pos, goal_pos, goal_id, holding, position, heading_yaw, gripper`). Adapt
  `WarehouseGymEnv` to that and pass it instead of `ToyPickupEnv`.
- Real policy: pass any `policy(obs) -> action(6,)` to `EvalHarness(policy=...)`.
- Training injection: add `CASlopeShaper.shaping(...)` to the staged reward in
  `env/warehouse_reward.py` (coordinate with P1 on where) — same module, torch tensors.
