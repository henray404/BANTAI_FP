# Experiment Harness — 2×2 Ablation + Baselines

**Owner:** P5 · **Created:** 2026-06-21 · **Budget:** 18 runs × 200k env steps

Factorial ablation **(CA-SLOPE on/off) × (Visual HER on/off)** on DreamerV3, plus two
model-free baselines (SAC, PPO) = **6 configurations × 3 seeds = 18 sequential runs**.

> **Step-by-step run procedure + preflight: [`RUNBOOK.md`](RUNBOOK.md).**
> Hyperparameters + paper mapping: [`hyperparameters.md`](hyperparameters.md).
> Reference list: [`../research/referensi.md`](../research/referensi.md).

---

## The six configurations

| # | config (`logname`) | stack | CA-SLOPE | Visual HER | isolates |
|---|--------------------|-------|----------|------------|----------|
| 1 | `c1_sac`             | SB3 SAC (model-free) | – | – | model-free floor (off-policy) |
| 2 | `c2_ppo`             | SB3 PPO (model-free) | – | – | model-free floor (on-policy) |
| 3 | `c3_dreamer_vanilla` | DreamerV3 (NM512)    | ✗ | ✗ | pure world-model effect |
| 4 | `c4_dreamer_caslope` | DreamerV3            | ✓ | ✗ | CA-SLOPE contribution |
| 5 | `c5_dreamer_her`     | DreamerV3            | ✗ | ✓ | Visual HER contribution |
| 6 | `c6_dreamer_full`    | DreamerV3            | ✓ | ✓ | combined (proposed) |

**Isolation logic** (pairwise comparisons reported by `analyze.py`):
`#6 vs #4` = pure Visual HER · `#6 vs #5` = pure CA-SLOPE · `#4,#5 vs #3` = each
component over the world model · `#3 vs #1,#2` = model-based vs model-free.

Single source of truth for the matrix: [`../../experiments/configs.py`](../../experiments/configs.py).

---

## How to run

```bash
conda activate isaaclab

# 1. Preview the full 18-run schedule (no training):
python -m experiments.run_all --dry-run

# 2. Smoke test one short run end-to-end (recommended before the full sweep):
python -m experiments.run_all --only 3 --seeds 0 --steps 5000

# 3. Run everything sequentially (resumable — re-run to continue after a crash):
python -m experiments.run_all

# 4. Aggregate results + significance tests:
python -m experiments.analyze --results training/results/ablation
```

Useful flags for `run_all`: `--only 3 4 5 6` (subset of configs), `--seeds 0`
(one seed), `--steps N`, `--timeout SECONDS`, `--no-headless`.

Each run is a **fresh subprocess** (one Isaac sim per process). A finished run writes a
`DONE` marker under its logdir; re-invoking `run_all` skips completed runs, so the sweep
is interruptible/resumable.

> **Blackwell / RTX 5050 caveat:** a finished Isaac run can leave a zombie `python.exe`
> (close() hang). `run_all` does not auto-kill processes; kill leftovers manually between
> runs, or use the Linux 2-GPU box (see `docs/setup/TRAINING_2GPU.md`). Pin the NVIDIA
> driver per `CLAUDE.md`.

---

## Tuning (edit one file)

All knobs live in [`../../experiments/ablation.yaml`](../../experiments/ablation.yaml):
training budget, eval cadence, CA-SLOPE mode/gains, Visual HER ratio, DreamerV3
`train_ratio`, and the SAC/PPO hyperparameters. Edit it, then pass `--config`:

```bash
python -m experiments.run_all --config experiments/ablation.yaml
```

`run_all` forwards `--config` to every run; `--seeds`/`--steps` on the CLI override the
YAML. Any key you omit falls back to the baked-in default in `experiments/settings.py`.
The structural matrix (which 6 configs exist) stays in `experiments/configs.py` — the YAML
is only the numbers.

CA-SLOPE `mode`: `category` (per-category gains — the RQ2 method), `generic` (single gain —
the RQ2 control), or `none`. Gains adopted from the teammate's `reward/ca_slope.py`
(`category_gains: [fragile, regular, heavy]`, `phase_b_offset` keeps Φ continuous at grasp).

---

## Outputs per run + reproducing the best

Each `training/results/ablation/<logname>_seed<n>/` contains:

| file | what |
|------|------|
| `eval_metrics.csv` | per-eval timeseries: `step, success_rate, success_std, mean_length, mean_return` |
| `run_config.yaml`  | exact resolved settings + flags (algo, seed, ca_slope, visual_her, all hyperparams) |
| `best/best.json`   | metrics + step of the best eval so far (highest success rate, ties by return) |
| `best/best_model*` | checkpoint at the best eval (SB3 `.zip`; DreamerV3 copy of `latest.pt`) |
| `best/run_config.yaml` | the config that produced the best model |
| `best/best_trajectory.csv` | best EPISODE's per-step actions + state (`step,a0..a5,robot_xyz,ee_xyz,holding,reward`) |
| `best/best_init.json` | scene snapshot at that episode's start (robot+box poses, goal) for replay |
| `DONE`             | resume marker written on success |

**Reproduce the best run:** read `best/run_config.yaml` for the exact hyperparameters and
re-launch with the matching `--config` / flags / seed; load `best/best_model*` for inference.

**Watch the best run in the GUI:**
```bash
python scripts/replay_best.py --run training/results/ablation/<logname>_seed<n>
```
Restores `best_init.json` then replays `best_trajectory.csv` action-by-action in Isaac Lab —
see the model's best decisions without retraining. (Snapshot restore is required because box
poses + goal are randomized each reset.)

---

## Metrics

Every run writes `training/results/ablation/<logname>_seed<n>/eval_metrics.csv`:

| column | meaning |
|--------|---------|
| `step` | env step at the eval |
| `success_rate` | fraction of eval episodes that delivered the correct-category box to the correct zone and released (`pickup_delivered`) |
| `mean_length` | mean steps to episode end |
| `mean_return` | mean episode return |

Eval cadence: **every 10 000 steps over 5 episodes** (spec). The same CSV format is
produced by all three stacks:
- **SAC/PPO** — an SB3 callback runs `experiments.metrics.evaluate_policy`.
- **DreamerV3** — `experiments.nm512_eval.EvalRecorder` wraps the NM512 eval env.

`analyze.py` reports the three spec metrics:
1. **success rate** — final-eval mean ± std across the 3 seeds.
2. **sample efficiency** — env steps to first reach a success-rate threshold (default 0.5).
3. **episode length** — mean steps to completion.

---

## Significance testing

Per-config success rate is reported as **mean ± std** over seeds {0,1,2}. Pairwise
comparisons use the **Mann–Whitney U test** (two-sided), chosen over a t-test because
n=3 seeds per config is too small to justify normality.

`analyze.py` uses `scipy.stats.mannwhitneyu` when scipy is installed, otherwise an **exact
permutation** null distribution (cheap for n=3). Caveat: with n=3 vs n=3 the smallest
achievable exact two-sided p is **0.1** — treat p-values as indicative and report effect
sizes (mean differences) alongside.

---

## Files

| path | role |
|------|------|
| `experiments/configs.py`     | 6-config registry + isolation comparisons + paper keys |
| `experiments/ablation.yaml`  | **editable knob file** (budget, CA-SLOPE gains/mode, HER, baselines) |
| `experiments/settings.py`    | loads `ablation.yaml` over baked-in defaults |
| `experiments/run_all.py`     | sequential 18-run orchestrator (resumable), forwards `--config` |
| `experiments/metrics.py`     | shared eval loop + `eval_metrics.csv` + `run_config.yaml` + `BestModelTracker` |
| `experiments/nm512_eval.py`  | DreamerV3 eval-env recorder → CSV + best snapshot |
| `experiments/analyze.py`     | aggregate + Mann–Whitney U → `summary.md` |
| `experiments/trajectory_recorder.py` | record best episode's actions → `best_trajectory.csv` + `best_init.json` |
| `env/scene_snapshot.py`      | capture/restore full scene state for faithful replay |
| `scripts/replay_best.py`     | replay the best episode in the Isaac Lab GUI |
| `reward/ca_slope.py`         | `CASlopeShaper` — category-aware PBRS potential (numpy+torch) |
| `reward/ca_slope_wrapper.py` | `CASlopeEnvWrapper` (mode = category/generic/none) |
| `env/her_nm512.py`           | Visual HER relabel + monkeypatch for the NM512 loop |
| `scripts/train_dreamer.py`   | DreamerV3 entry (`--ca_slope`, `--visual_her`, `--config`) |
| `scripts/train_sac.py`       | SAC/PPO entry (`--algo`, `--ca_slope`, `--config`) |
| `tests/test_experiments.py`  | pure unit tests (no Isaac) |
| `tests/test_ca_slope.py`     | CA-SLOPE unit tests (teammate's, runs without Isaac) |

---

## Status

- ✅ Harness, wiring, stats, config loader, CSV/best-model, unit tests (30 passing, no Isaac).
- ✅ CA-SLOPE = teammate's `reward/ca_slope.py` (per-category gains; my weaker wrapper dropped).
- ✅ Physics: base drive effort capped to stop the chassis clipping through racks
  (`bugs_errors/2026-06-21_base-clips-through-racks.md`) — **UNVERIFIED on hardware**.
- ⚠️ **End-to-end training UNVERIFIED on hardware** — gated by the Blackwell camera/close
  issues on the RTX 5050. Run on the Linux 2-GPU box. The non-Isaac logic
  (configs, settings, stats, HER relabel, CA-SLOPE) is unit-tested and green.
