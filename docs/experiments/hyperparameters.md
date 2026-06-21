# Hyperparameters & Paper Mapping

**Created:** 2026-06-21 · Companion to [`README.md`](README.md).

Every value below is grounded in a **source already in the repo** (vendored config or
baseline module) — no new values were invented. Reference numbers (`#N`) point at
[`../research/referensi.md`](../research/referensi.md).

> **Live source:** the tunable values are now in
> [`../../experiments/ablation.yaml`](../../experiments/ablation.yaml) (loaded by
> `experiments/settings.py`). Edit that file and pass `--config` — this doc explains the
> values; the YAML sets them.

---

## Shared experiment budget (all 6 configs)

| param | value | source |
|-------|-------|--------|
| env steps / run | 200 000 | spec; `experiments/configs.TOTAL_STEPS` |
| seeds | {0, 1, 2} | spec; `configs.SEEDS` |
| eval cadence | every 10 000 steps, 5 episodes | spec; `configs.EVAL_EVERY/EVAL_EPISODES` |
| parallel envs | 1 | 8 GB VRAM + active arm IK (`CLAUDE.md`) |
| significance test | Mann–Whitney U (two-sided) | spec; small n=3 |

---

## Configs #3–6 — DreamerV3 (NM512 `dreamerv3-torch`)

DreamerV3's design claim is **fixed, task-agnostic hyperparameters** — so configs #3–6
use the vendored paper defaults unchanged (`models/dreamerv3/vendor/configs.yaml`
`defaults:`), with only the warehouse task overrides in `models/dreamerv3/config.py`.
Paper: DreamerV3 **#1** (Hafner et al. 2023); robot precedent DayDreamer **#4**; impl
NM512 (`E. GitHub Repos`).

**World model (RSSM):**

| param | value |
|-------|-------|
| `dyn_deter` (h) | 512 |
| `dyn_stoch` × `dyn_discrete` | 32 × 32 |
| RSSM feature dim | 512 + 32·32 = **1536** |
| `units`, `dyn_hidden` | 512 |
| activation / norm | SiLU / RMS-style norm on |
| `kl_free`, `dyn_scale`, `rep_scale` | 1.0, 0.5, 0.1 |
| `model_lr`, `grad_clip` | 1e-4, 1000 |

**Actor–critic (in imagination):**

| param | value |
|-------|-------|
| actor/critic lr | 3e-5 |
| actor entropy | 3e-4 |
| critic | symlog two-hot, slow target (EMA frac 0.02) |
| `discount` (γ) | 0.997 |
| `discount_lambda` | 0.95 |
| `imag_horizon` | 15 |
| `batch_size` × `batch_length` | 16 × 64 |
| `train_ratio` | 512 |

**Warehouse task overrides** (`config.py` `WAREHOUSE_OVERRIDES`): `action_repeat=1`
(env already decimates 200→10 Hz), `time_limit=1000`, `prefill=2000`, `size=[64,64]`,
encoder/decoder `cnn_keys="image"`, `mlp_keys` = the 8 low-dim obs keys, `compile=False`
(Windows). `goal_id` (3-dim one-hot) feeds the RSSM directly — **no CLIP projection**
(removed 2026-06-08).

> **Wall-clock note:** `train_ratio=512` is DreamerV3's sample-efficiency default (512
> gradient updates per env step). It is the right setting for the fixed 200k-env-step
> budget but is compute-heavy on one GPU. If wall-clock is the binding constraint,
> lower it via an override — but keep it identical across configs #3–6 or the ablation
> comparison breaks.

---

## Config #4 / #6 — CA-SLOPE (Category-Aware SLOPE)

Potential-based reward shaping added to the env reward: `F = γ·Φ(s') − Φ(s)`. By the PBRS
theorem the optimal policy is unchanged. Pure-DL: Φ reads env state selected by `goal_id`
(target box + matching zone), **not** a vision detector. Canonical code (adopted from a
teammate's branch 2026-06-21): `reward/ca_slope.py` (`CASlopeShaper`) +
`reward/ca_slope_wrapper.py` (`CASlopeEnvWrapper`). It is backend-agnostic (same module in
the torch training reward and the numpy headless eval). Set via `ablation.yaml: ca_slope`.

The potential is a category-weighted remaining-distance to task completion:

| phase | Φ(s) |
|-------|------|
| not holding | `−gain · (dist(ee, box) + phase_b_offset)` |
| holding | `−gain · dist_xy(box, zone)` |

`phase_b_offset` (≈13 m) keeps Φ continuous when `holding` flips at grasp (so F rewards the
grasp instead of punishing it). Terminal convention Φ(terminal)=0 → F = −Φ(s).

| param | value | meaning |
|-------|-------|---------|
| `mode` | `category` | per-category gains (RQ2 method); `generic` = single gain (control); `none` = off |
| γ | 0.997 | match the DreamerV3 discount (PBRS exact only here) |
| `category_gains` | `[1.0, 1.5, 2.0]` | gain per [fragile, regular, heavy] — heavier = steeper guidance |
| `generic_gain` | 1.5 | the single gain used when `mode=generic` |
| `phase_b_offset` | 13.0 | continuity offset across the grasp |

RQ2 control: `mode=category` vs `mode=generic` isolates whether *per-category* gains help
beyond a single generic gain (same code path, one flag).

Papers: PBRS **#15** (Ng, Harada, Russell 1999) — optimality preservation; dynamic PBRS
**#16** (Devlin & Kudenko 2012) — varying potential across phases.

---

## Config #5 / #6 — Visual HER

Relabel failed-but-grasped episodes: set `goal`/`goal_id` to the zone the robot got
closest to after grasping; give terminal `success_reward`. For DreamerV3 the relabeled
episode is injected into the NM512 train cache (`env/her_nm512.py`); the custom buffer
path uses `buffer/visual_her.py`.

| param | value |
|-------|-------|
| `her_ratio` | 0.5 |
| `success_reward` | 10.0 (matches env delivery reward) |
| strategy | achieved-zone relabel + terminal success |

Papers: HER **#11** (Andrychowicz et al. 2017); visual HER **#12** (Sahni et al. 2019);
RIG **#13** (Nair et al. 2018); GCRL survey **#14**.

---

## Configs #1 / #2 — model-free baselines (Stable-Baselines3)

Source: `training/baselines/sac.py`, `training/baselines/ppo.py` (`DEFAULTS`).
`MultiInputPolicy` over the Dict obs (uint8 pixels → NatureCNN + vector keys).

**SAC (#1)** — off-policy. Paper **#18** (Haarnoja et al. 2018).

| param | value |
|-------|-------|
| learning rate | 3e-4 |
| buffer size | 200 000 |
| learning starts | 5 000 |
| batch size | 256 |
| τ | 0.005 |
| γ | 0.99 |
| train_freq / grad steps | 1 / 1 |

**PPO (#2)** — on-policy. Paper **#19** (Schulman et al. 2017).

| param | value |
|-------|-------|
| learning rate | 3e-4 |
| n_steps | 2048 |
| batch size | 256 |
| n_epochs | 10 |
| γ / GAE-λ | 0.99 / 0.95 |
| clip range | 0.2 |
| entropy coef | 0.0 |

Baseline repo: Stable-Baselines3 (`DLR-RM/stable-baselines3`), referensi `E. GitHub Repos`.

---

## Why these comparisons (recap)

| comparison | conclusion it supports |
|------------|------------------------|
| #3 vs #1,#2 | does the world model beat model-free? |
| #4 vs #3 | does CA-SLOPE help on top of the world model? |
| #5 vs #3 | does Visual HER help on top of the world model? |
| #6 vs #4 | pure Visual HER effect (CA-SLOPE held on) |
| #6 vs #5 | pure CA-SLOPE effect (HER held on) |
