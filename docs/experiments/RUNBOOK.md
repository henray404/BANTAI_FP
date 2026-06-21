# Training Runbook — Step by Step

**Goal:** run the 18-run ablation (6 configs × 3 seeds) end to end.
**Companions:** [`README.md`](README.md) (what each piece is), [`hyperparameters.md`](hyperparameters.md)
(the knobs), [`../setup/INSTALL_LINUX.md`](../setup/INSTALL_LINUX.md) +
[`../setup/TRAINING_2GPU.md`](../setup/TRAINING_2GPU.md) (machine setup).

> Read **§0 Before you start** first. Most failures come from skipping a preflight check.

---

## 0. Before you start — what to check FIRST

| # | Check | Command / action | Why |
|---|-------|------------------|-----|
| 1 | Right machine | Linux 2-GPU box, NOT the RTX 5050 | Blackwell camera/close-hang blocks the full env (`bugs_errors/2026-05-22_sdp-camera-crash-blackwell.md`). |
| 2 | Conda env | `conda activate isaaclab` | All deps (Isaac 5.1, torch 2.7+cu128, sb3, ruamel) live here. |
| 3 | GPU free | `nvidia-smi` — VRAM mostly free, no zombie `python.exe` | A leftover Isaac process holds VRAM; one env + arm IK already saturates 8 GB. |
| 4 | Driver pinned | not 591.x/595.x on Blackwell | Those reintroduce the camera crash. |
| 5 | Sim actually runs | `python tests/test_env.py --num_envs 1` → 10/10 PASS | Confirms camera + env work before burning a long run. |
| 6 | Assets present | `assets/` synced (scp from the lab box if missing) | Scene USDs must resolve. |
| 7 | Disk space | ~a few GB free under `training/results/` | Each run writes episodes + checkpoints + CSV. |

**Hard rules**
- One GPU drives ONE run at a time — the 18 runs are **sequential**. `run_all` enforces this.
- After each Isaac run on Blackwell, kill the zombie `python.exe` before the next (the
  orchestrator does NOT auto-kill — it would kill itself). On Linux usually not needed.
- Keep `train_ratio` identical across configs #3–6, or the ablation comparison is invalid.

---

## 1. Configure the run (edit ONE file)

All knobs are in [`../../experiments/ablation.yaml`](../../experiments/ablation.yaml):
budget, eval cadence, CA-SLOPE mode/gains, Visual HER ratio, `train_ratio`, SAC/PPO params.
Edit it, save. Nothing else to touch (the 6-config matrix is fixed in `experiments/configs.py`).

```bash
# inspect / edit
nano experiments/ablation.yaml
```

---

## 2. Dry-run — see the 18 commands WITHOUT training

```bash
python -m experiments.run_all --dry-run --config experiments/ablation.yaml
```
Confirm: 18 runs, correct flags (`--ca_slope` on #4/#6, `--visual_her` on #5/#6), steps =
your YAML value. If this looks wrong, fix the YAML — do not proceed.

---

## 3. Smoke test — ONE short real run end to end

Before committing days of compute, prove a real run works with a tiny budget:

```bash
python -m experiments.run_all --only 3 --seeds 0 --steps 5000 \
       --config experiments/ablation.yaml
```
Check it produced:
```
training/results/ablation/c3_dreamer_vanilla_seed0/
  eval_metrics.csv     # has at least one row
  run_config.yaml      # records what ran
  best/best.json       # best snapshot written
  DONE                 # success marker
```
If it crashed: see §6. **Do not start the full sweep until the smoke run is clean.**

Optional smoke for a baseline too (different stack):
```bash
python -m experiments.run_all --only 1 --seeds 0 --steps 5000 --config experiments/ablation.yaml
```

---

## 4. Full sweep — all 18 runs

```bash
python -m experiments.run_all --config experiments/ablation.yaml
```
- **Resumable:** a finished run writes `DONE`; re-running skips it. Safe to Ctrl-C and restart.
- **Long:** `train_ratio=512` (DreamerV3 default) is compute-heavy. Estimate wall-clock from
  the smoke run's FPS × 200 000 steps × 18 — if too long, lower `train_ratio` in the YAML
  (keep it equal across #3–6) and re-smoke.
- Run detached so an SSH drop doesn't kill it:
  ```bash
  nohup python -m experiments.run_all --config experiments/ablation.yaml \
        > training/results/ablation/run_all.log 2>&1 &
  tail -f training/results/ablation/run_all.log
  ```
- Per-run timeout (kills a hung run, marks it failed, moves on):
  ```bash
  python -m experiments.run_all --config experiments/ablation.yaml --timeout 36000
  ```

Subsets while iterating:
```bash
python -m experiments.run_all --only 3 4 5 6      # DreamerV3 configs only
python -m experiments.run_all --seeds 0 1         # fewer seeds
```

---

## 5. Monitor while it runs

- **Live CSV:** `tail -f training/results/ablation/c6_dreamer_full_seed0/eval_metrics.csv`
  → `success_rate` should trend up every 10k steps.
- **Best so far:** `cat training/results/ablation/<run>/best/best.json`.
- **W&B** (baselines, if `--wandb` + installed): project `bantai-warehouse`.
- **GPU:** `watch -n5 nvidia-smi`.

---

## 6. If a run fails — quick triage

| Symptom | Likely cause / fix |
|---------|--------------------|
| Camera SDP crash / hang at startup | Wrong machine or driver (Blackwell). Use the Linux box; check §0.1/0.4. |
| `ModuleNotFoundError` (reward/experiments/...) | Run from repo root with `isaaclab` env active. |
| OOM / VRAM | Kill zombie `python.exe`; ensure `num_envs=1`; close other GPU jobs. |
| Robot drives through racks | Known — base drive capped already; tune `BASE_DRIVE_EFFORT` (`bugs_errors/2026-06-21_base-clips-through-racks.md`). |
| Run hangs at end (close() hang) | Blackwell only; kill the leftover `python.exe`, then re-run (it resumes via `DONE`). |
| sb3 not installed | `pip install -r requirements-ml.txt` (keep it from downgrading torch). |

A failed run does NOT get a `DONE` marker — just re-run `run_all`, it retries only the missing ones.

---

## 7. Aggregate + report

```bash
python -m experiments.analyze --results training/results/ablation
```
Writes `training/results/ablation/summary.md`: per-config success rate (mean ± std over seeds),
sample efficiency (steps to 50% success), and the Mann–Whitney U p-values for the isolation
comparisons (#6vs#4, #6vs#5, #4/#5vs#3, #3vs#1/#2).

> n=3 seeds → smallest exact two-sided p is 0.1. Report effect sizes (mean diffs) alongside p.

---

## 8. Reproduce the best run later

Each run's best checkpoint + the exact config that made it are saved:
```
training/results/ablation/<run>/best/
  best.json          # step + metrics
  best_model.zip|pt  # checkpoint (SB3 .zip / DreamerV3 latest.pt copy)
  run_config.yaml    # the exact settings + flags
```
Read `run_config.yaml`, relaunch with the matching `--config`/seed/flags, and load
`best_model*` for inference.

---

## TL;DR

```bash
conda activate isaaclab
python tests/test_env.py --num_envs 1                      # 0. sim works?
nano experiments/ablation.yaml                             # 1. set knobs
python -m experiments.run_all --dry-run --config experiments/ablation.yaml   # 2. preview
python -m experiments.run_all --only 3 --seeds 0 --steps 5000 --config experiments/ablation.yaml  # 3. smoke
python -m experiments.run_all --config experiments/ablation.yaml             # 4. full 18
python -m experiments.analyze --results training/results/ablation            # 7. report
```
