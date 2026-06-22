# Training Readiness — Jalan A (arm frozen in training, active in teleop)

**Date:** 2026-06-22
**Decision:** Jalan A — the Franka arm is **frozen during training** (policy learns base + grasp-proximity
only, stable, no drift/flail) and **active only in teleop** (`drive_env`, `arm_active=True`). This
de-risks training: the magnetic/proximity grasp is the already-proven path (demo_pickup carried a box
end-to-end), so arm IK tuning is **not** a blocker for training.

This doc = (1) gap analysis, (2) ordered steps for you, (3) flaws/footguns to watch.

---

## ✅ Fixes applied this pass (2026-06-22 code pass — by Claude)

Code/config fixes that needed no Isaac run (verified standalone). What's done vs still-on-you:

| Fix | What changed | Verified |
|---|---|---|
| **C1 — dead approach gradient (CRITICAL)** | Added `self.ee_pos_world` (env-local world) in `warehouse_env`; `reward_pickup.approach_box_distance` now uses it instead of the base-frame `ee_pos`. Distance now **shrinks on approach** (was constant → no gradient). Updated `tests/test_reward_pickup.py`. | ✅ 6/6 tests pass + asserted `far=3.0 → near=0.5` shrinks |
| **C2 / flaw-6 — `time_pen` suicide incentive** | `configs/reward_weights.yaml`: `time_pen -0.05 → -0.005` (with a warning comment). | ✅ yaml loads |
| **flaw-1 — reward triple-source drift (2 of 3)** | `configs/env_config.yaml` `reward:` section now banners **⚠️ NOT LOADED — see reward_weights.yaml**. `reward_weights.yaml` is the single live source. | ✅ |
| **Jalan A wiring** | `WarehouseGymEnv(arm_active=False)` default → arm frozen for ALL training entries; `drive_env` passes `arm_active=True`. | ✅ parses, wiring confirmed |

**Could NOT fix (and why):**
- **flaw-1, 3rd source:** `CLAUDE.md` reward block is still stale — that file is **write-protected** for agents. *You* must update it manually (point it at `reward_weights.yaml`).
- **B1/B2/B3, C3 magnitudes, home-pose retune:** need a live Isaac run or a tuning judgment call — see steps below. C1 being fixed is the prerequisite that makes a Step-4 sanity run meaningful now.

---

## 1. Gap analysis — what is / isn't ready

### ✅ Done (verified this session, standalone — NOT in a live Isaac run)
| Item | State |
|---|---|
| Import fix `models.WorldModel` collision | Fixed + repro'd. Was the crash that stopped the last run. |
| `--stage` flag (curriculum, fixed) | Wired into `train_dreamer.py`, parses. |
| Jalan A arm freeze flag (`arm_active`) | Wired: training frozen (default), `drive_env` active. |
| Reward/penalty weights → `configs/reward_weights.yaml` | Live-loaded, + `idle_slow` (30s) penalty. |
| Env physics → `configs/env_config.yaml` (speed/episode/decimation/dt) | Live-loaded. |
| Arm home pose → `configs/env_config.yaml` (`robot.home_joint_pos`) | Live-loaded. |
| Teleop sensitivity → `configs/teleop.yaml` (`ee_sens` etc.) | Live-loaded. |
| DreamerV3 integration (obs adapter, HER, CA-SLOPE, eval recorder, best-model, traj) | Code complete (P2/P3/P5). |

### ⚠️ NOT verified — blockers before a long training run
| # | Gap | Why it blocks | Severity |
|---|---|---|---|
| **B1** | **End-to-end DreamerV3 run never confirmed on RTX 5050 (Blackwell)** | Last run died at `WorldModel` build (now fixed). STATUS.md "verified" was on a *different* machine (RTX 2080 Ti). Downstream errors (obs shape, encoder keys, camera) may still surface. | **HIGH** |
| **B2** | **Grasp-proximity works with frozen arm + current home pose** | Grasp fires on hand↔box proximity. Frozen hand sits at a fixed offset; if it's awkward vs the box, grasp rarely fires → no `grasp +5` → unlearnable. Demo proved carry works, but re-confirm after the home-pose/config churn. | **HIGH** |
| **B3** | **Camera works on this machine** | CLAUDE.md says the Blackwell SDP crash was fixed by driver downgrade to 580.88. `pixels` obs depends on it. Confirm the camera still renders after recent changes. | **MED** |

### 🔧 Not tuned (not blockers, but needed for good learning)
| Item | Current | Note |
|---|---|---|
| Reward weights | defaults in `reward_weights.yaml` | `time_pen` was bumped to **-0.05** (10× too high — likely makes the policy "give up"). Recommend back to -0.005. |
| Curriculum stage | default `--stage 3` (full chain, sparse) | Start `--stage 2` (grasp isolation) for dense signal. |
| `ee_sens` (teleop) | 1.3 | fine for teleop; not training-relevant. |

### ❌ Missing / by-design-absent
| Item | Status | Priority |
|---|---|---|
| **Curriculum auto-advance** | Not in the DreamerV3 path. `--stage` is fixed; you advance manually (stop → change `--stage` → restart). The success-gate exists only in `policy/train_loop.py` (SAC/PPO), unused by dreamer. | MED — train works, just manual. |
| **Batch eval + success-rate aggregation** | Partial (`EvalRecorder` logs a CSV; aggregation thin). | LOW |
| **Multi-env** | Fixed at `num_envs=1` (8GB VRAM). Wall-clock slow (~10 env-steps/s). | Known limit. |
| **W&B** | Logger ready, not installed/login. stdout fallback works. | LOW |
| Perception (YOLO/CLIP) | Removed by design (pure-DL, `goal_id` one-hot). | N/A |

---

## 2. Steps for you (in order — don't skip)

### Step 0 — Commit current WIP first
A lot changed uncommitted this session. Checkpoint before running so a bad experiment is revertible.
```bash
git add -A && git commit -m "wip: import fix, --stage, config yamls, Jalan A arm freeze"
```

### Step 1 — Confirm training STARTS (B1)  ⟵ the real blocker
```bash
python scripts/train_dreamer.py --num_envs 1 --headless --stage 2
```
- **Pass:** you see `[curriculum] stage fixed at 2`, scene builds, prefill runs, then `train` steps begin (no `WorldModel` / shape / key error).
- **Fail:** copy the traceback here. The import fix handles `WorldModel`; a *different* error means a new downstream gap.

### Step 2 — Confirm grasp can fire with frozen arm (B2)
Teleop, arm active, drive the BASE to a box and close the gripper (you do NOT need the arm for the grasp — it's proximity):
```bash
python scripts/drive_env.py
```
- Drive base up to a box (arrows), press **K** to close gripper.
- **Pass:** `holding=1` prints. Grasp-proximity works → training can learn it.
- **Fail:** `holding` stays 0 no matter how close → grasp threshold / hand offset is wrong. Flag it; we fix `grasp` proximity radius or the frozen home offset.
- (Optional) `--cam 30` to dump onboard camera frames → confirms B3 (camera renders).

### Step 3 — Fix the obvious reward footgun
Edit `configs/reward_weights.yaml`:
```yaml
time_pen:  -0.005   # was -0.05 (10× too punishing)
approach:  -0.05    # stronger pull toward the box
collision:  1.0     # softer (was 2.0) so it dares to move
```

### Step 4 — Short training sanity run (watch the signal)
```bash
python scripts/train_dreamer.py --num_envs 1 --headless --stage 2 --steps 20000
```
Watch in the logs (`training/results/dreamerv3/`):
| Signal | Healthy | If not |
|---|---|---|
| `grasp` reward | fires > 0 within ~5k steps post-prefill | stays 0 → back to Step 2 (mechanics) |
| episode return | trends up | flat → reward/stage wrong |

### Step 5 — Scale up + advance the curriculum manually
- Grasp reliable at stage 2 → stop → restart `--stage 3` (full chain) → continue.
- Then long run with full `--steps`.

---

## 3. Flaws / footguns found

1. **Reward config TRIPLE-source drift (real footgun).**  🟡 2 of 3 fixed (env_config bannered;
   CLAUDE.md still stale — write-protected, fix manually). Three places define reward weights and they
   **disagree**:
   - `configs/reward_weights.yaml` — **LIVE** (what actually runs): approach -0.02, collision 2.0.
   - `configs/env_config.yaml` `reward:` section — **dead doc** (NOT loaded): approach -0.01, collision 5.0.
   - `CLAUDE.md` reward block — stale: approach -0.01, collision -5.
   Editing the wrong one = no effect, silently. **Fix:** delete the `reward:` + `termination:` reward
   blocks from `env_config.yaml` (or add a `# NOT LOADED — see reward_weights.yaml` banner), and update
   the CLAUDE.md block. Only `reward_weights.yaml` is live.

2. **`obs.ee_pos` is a WORLD-frame delta, not base-frame.** `warehouse_env.py` computes
   `body_pos_w[ee] - body_pos_w[base]` (world axes), but CLAUDE.md + `env_config.yaml` call it
   "base frame". Minor for frozen-arm training (ee_pos is ~constant), but the obs contract doc is wrong.

3. **STATUS.md is optimistic + machine-mismatched.** It marks the whole pipeline "✅ DONE / verified
   2026-06-20" — but that was on RTX 2080 Ti, and the DreamerV3 entry never ran end-to-end on the
   RTX 5050. Treat STATUS.md as component-level intent, not end-to-end proof on your hardware.

4. **Curriculum has TWO meanings in this repo.** `env/curriculum.py` (our `--stage` mechanism) vs the
   Isaac `CurriculumManager` ("0 active terms" you saw — that's normal, we don't use Isaac's curriculum
   term API). Don't chase the Isaac one.

5. **No auto-advance in the dreamer path.** You must babysit stage transitions manually. If a long
   unattended run is wanted, that's a feature to add (port the success-gate from `policy/train_loop.py`).

6. **`time_pen -0.05`** ✅ FIXED → reverted to -0.005 in `reward_weights.yaml`. (Was: you set it.)
   Over a 1000-step episode -0.05 was -50,
   dwarfing `grasp +5` / `deliver +10` → the policy is incentivized to end episodes fast (crash / leave
   bounds) rather than do the task. Strongly recommend -0.005.

---

## TL;DR
- **Jalan A is wired.** Training = arm frozen (stable, proven grasp-proximity). Teleop = arm active.
- **The one true blocker is B1:** run `train_dreamer.py --stage 2` and confirm it starts. Everything
  else is tuning or nice-to-have.
- **Fix the reward triple-source drift** so you're not tuning a dead file.
- **Drop `time_pen` back to -0.005.**

---

# 4. Task division — parallel work across laptops

Split by **machine capability**: Track A needs the Isaac/GPU laptop (RTX 5050) and physically runs the
sim; Track B is pure code/config/doc, runs + unit-tests on ANY laptop with no Isaac, then merges and is
verified on Track A. Work both tracks at once.

## Track A — Isaac/GPU laptop ONLY (sim required)
| ID | Task | Depends on | Done when |
|----|------|-----------|-----------|
| A1 | **Run `train_dreamer.py --stage 2` end-to-end** (B1) | the C1 fix (done) | reaches `train` steps, no crash; paste any traceback |
| A2 | **Teleop grasp check** (B2): drive base to box, `K`, see `holding=1` | — | grasp fires reliably; note the standoff that works |
| A3 | **Camera render check** (B3): `drive_env --cam 30`, inspect `_cam_debug/` | — | onboard frames look right |
| A4 | **Home-pose retune** (`env_config robot.home_joint_pos`) IF A2 shows the hand can't reach | A2 | hand sits over the front-floor box |
| A5 | **Sanity run + watch metrics** (`--steps 20000`): `grasp` reward > 0, return trends up | A1, reward fixes | grasp fires within ~5k post-prefill |
| A6 | **W&B install + login** (optional) | — | dashboard logs, or keep stdout |

## Track B — ANY laptop (no Isaac; unit-test locally, verify later on A)
| ID | Task | File(s) | Testable w/o Isaac |
|----|------|---------|--------------------|
| B-1 | **CLAUDE.md reward block** → point at `reward_weights.yaml` (the 3rd drift source; agent-protected so a human does it) | `CLAUDE.md` | n/a (doc) |
| B-2 | **C3 reward-shaping redesign**: potential-based shaping (`F=γΦ(s')−Φ(s)`, `Φ=−dist`, zero at reset) + rescale terminal bonuses ≥10× max dense | `reward_pickup.py`, `warehouse_reward.py`, `reward_weights.yaml` | ✅ pure-tensor unit tests |
| B-3 | **Curriculum auto-advance for the dreamer path**: port the success-gate from `policy/train_loop.py` into a callback the vendor loop can call | `scripts/train_dreamer.py` (+ small helper) | ✅ logic unit-testable |
| B-4 | **Wire remaining env knobs to yaml**: spawn ranges, `STUCK_STEPS`, bounds, actuators (extend `_load_env_config`) | `warehouse_env.py`, `warehouse_scene.py`, `env_config.yaml` | ✅ loader unit-testable |
| B-5 | **Batch eval + success-rate aggregation** (currently thin) | `experiments/nm512_eval.py`, `experiments/metrics.py` | ✅ on recorded CSVs |

**Merge protocol:** Track B pushes small PRs/branches; Track A pulls and runs the sim verification.
Only A can sign off "works" — B can only prove "logic correct + tests pass."

---

# AUDIT ADDENDUM — deeper wiring pass + run metrics + references (2026-06-22, second pass)

This addendum supersedes parts of §3 above. B1 is now MOOT: the pipeline **does run end-to-end** —
`training/results/dreamerv3/metrics.jsonl` has a real 14-episode / ~11k-step run with `image_loss`
logged, so camera + obs + WorldModel all work on this machine. The new question the metrics answer:
**why does return never go positive** (best `train_return` ≈ −18, most −60…−185). Three findings,
the first one corrects §3-flaw-#2.

## C1 (CRITICAL) — the approach reward is frame-mismatched → Phase-A dense gradient is DEAD  ✅ FIXED 2026-06-22
> **FIXED:** `ee_pos_world` added + `approach_box_distance` switched to it (see "Fixes applied" near
> the top). The analysis below is kept as the rationale.

§3 flaw #2 called the `ee_pos` world/base-frame issue "minor for frozen-arm training." **It is not
minor — it is the blocker.** The *reward* consumes it:
- `reward_pickup.approach_box_distance` = `norm(env.ee_pos − env.box_pos)`.
- `env.ee_pos` (set in `update_grasp`) = `body_pos_w[ee] − body_pos_w[base]` → ~constant
  `(-0.93,-0.32,0.96)` with the arm frozen (matches teleop debug `ee_pos=[-0.928,-0.315,0.959]`).
- `env.box_pos` = `root_pos_w[box] − env_origins` → env-local world, e.g. `(3.0,6.0,0.16)`.

Different frames. As the robot drives at the box, `box_pos` is fixed and `ee_pos` only *rotates*
(constant magnitude), so **the distance does not shrink on approach** → the −0.02·dist term gives no
gradient toward the box. The grasp *detector* is fine (it uses `ee_world` vs `box_world`); only the
*reward* reuses the mismatched buffers. **Fix:** shape on chassis↔box (`position` vs `box_pos`, both
env-local — already in obs), or add `self.ee_pos_world = body_pos_w[ee] − env_origins` and shape on
that. Keep the base-frame `ee_pos` *obs* (proprioception is fine).

**Run-metrics proof:** `holding_loss ≈ 0.001` every step (the `holding` bit is ~always 0 → grasp
essentially never fires); return never positive. Compounded by C3 below (frozen tucked hand sits ~1 m
behind+up the base, so even with a working approach signal the magnetic grab is geometrically
near-unreachable on a front-floor box — apply env_config's "forward-down try" home angles).

## C2 (HIGH) — early terminations confirm the `time_pen` suicide-incentive
`out_of_bounds`/`stuck` are terminal (`is_terminal=True` → discount 0, no bootstrap). With
`time_pen −0.05` × 1000 = −50 cumulative vs best-case task reward +15, ending the episode early is
value-optimal for a not-yet-competent policy. Metrics show it happening: episodes truncate at
`train_length` 538/639/641/689/719/726/943 instead of the full 1000. Confirms §3 item 6 with evidence.

## C3 (refinement) — terminal bonuses are also UNDER-scaled for the 1000-step horizon
Separate from time_pen: the reward-shaping literature's "dominance rule" says the terminal success
bonus should be **≥10× the max cumulative dense shaping per episode**. At 1000 steps your dense +
time terms accumulate to tens–hundreds, while `grasp +5` / `deliver +10` are one-shot. So even after
fixing C1, raise terminal bonuses (≈50–100) and/or shrink/normalize dense weights into [−1,1].
(Ng/Russell potential-based shaping `F=γΦ(s')−Φ(s)`, `Φ=−dist`, is the rigorous way to keep the
optimal policy invariant — and zero Φ at reset to avoid a step-1 spike.)

## Good news — these ARE wired correctly (verified this pass)
- **`mlp_keys` captures all 8 low-dim obs** (`config.py _MLP_KEYS`). The classic DreamerV3 trap
  (vision preset's `mlp_keys:'$^'` silently dropping proprioception) is **avoided** — metrics show
  `box_pos_loss`/`ee_pos_loss`/`goal_id_loss` all being modelled. Good.
- **IK uses `ik_method="dls"`** (damped least squares) — the recommended robust choice near
  singularities (not `pinv`). Good (only matters once the arm is unfrozen / for teleop).
- Camera + contact sensor on the moving `base_link`, base-drive yaw projection, floating-base weld,
  arm-gravity-disable — all correct (see §1 ✅ and the env code comments).

## VRAM / budget note (8 GB RTX 5050)
num_envs=1 is **on-design** for DreamerV3 (the paper runs one GPU, one env; sample-count won't be the
problem). Your real limits are **wall-clock** (`fps≈1` in metrics → 200k steps ≈ ~2 days/run) and
**8 GB VRAM**. If you hit OOM, use the danijar/NM512 `small`/`medium` size block rather than the
default. Don't quote the "200M steps / 8 h" figures from PPO/4096-env blogs — those don't transfer.

---

## References (web-research pass, 2026-06-22)
Confidence: [H]=primary (paper/official repo), [M]=reputable secondary, [L]=vendor blog/weak match.

**DreamerV3 hyperparameters & config** [H]
- Hafner et al., "Mastering Diverse Domains through World Models," arXiv:2301.04104.
- Official: github.com/danijar/dreamerv3 · Your port: github.com/NM512/dreamerv3-torch (`configs.yaml`).
- NM512 defaults: `batch_size 16`, `batch_length 64`, `train_ratio 512`, `model_lr 1e-4`,
  actor/critic `lr 3e-5`, RSSM `dyn_deter 512 / dyn_stoch 32×32`, `kl_free 1.0`, `dyn_scale 0.5`,
  `rep_scale 0.1`, `imag_horizon 15`, `discount 0.997`, actor `entropy 3e-4`, `prefill 2500`.
- ⚠️ `train_ratio` 512 (NM512) vs 32 (danijar example) use **different denominators** — verify which
  your loop uses before copying. Your `experiments/settings.py` sets `train_ratio 512`, `prefill 2000`.

**Image + low-dim obs handling** [H]
- DreamerV3 uses **symlog** on encoder inputs + vector decoder (`vector_dist: symlog_mse`) — you do
  NOT need to hand-normalize the low-dim keys. Pixel-recon can dominate loss (12288 img dims vs ~17
  low-dim) → watch per-key `box_pos`/`ee_pos` decoder loss. vitalab.github.io DreamerV3 summary.

**Reward shaping / dense-vs-sparse** [M/L + one H]
- Ng, Harada, Russell (1999) "Policy invariance under reward transformations" — potential-based
  shaping theorem [H].
- Dominance rule, phase gating, anti-hack heuristics: roboticscenter.ai/blog/reward-shaping-robot-
  manipulation [L, vendor] + claru.ai/guides/how-to-design-a-reward-function-for-robot-learning [L].
- arXiv:2206.02462 (RL pick-place reward shaping + curriculum, documents a real reward-hack) [M];
  Dense2Sparse arXiv:2003.02740 (switch dense→sparse mid-training) [M].

**Curriculum (reach→grasp→place)** [H/M]
- Breyer et al., "Comparing Task Simplifications to Learn Closed-Loop Object Picking," arXiv:1803.04996
  — workspace/spawn-distance curriculum, wrist cam + EE-displacement actions (closest analog to you) [H].
- MDPI Biomimetics 8(2):240 (task decomposition) [M]; KCAC arXiv:2505.10522 (cross-task transition
  point) [M]. Heuristic: advance on success ~50–70% and on **physical** signals, not step counts [L].

**Visual HER** — recognized line, not one canonical name [H]
- RIG "Visual RL with Imagined Goals," Nair et al. NeurIPS 2018 — relabel goals in **VAE latent**
  (best citation for "Visual HER").
- HALGAN arXiv:1901.11529 — literally titles it "visual hindsight experience replay."
- LEXA (orybkin.github.io/lexa) + MUN/"GC-Dreamer" NeurIPS 2024 — HER-style relabeling inside a
  **Dreamer** world model. MHER arXiv:2107.00306 — model-based HER.

**Differential IK for RL** [H]
- Isaac Lab `DifferentialIKController` docs + source; use `dls` (you do). Issue #4825: diff-IK does
  not respect joint limits / gravity-compensate under a payload → expect EE sag while carrying.
- ⚠️ **NO SOLID REFERENCE** for an exact per-step EE delta at 10 Hz — copy `action_scale` from
  `Isaac-Lift-Cube-Franka-v0` instead of guessing. (Your `EE_STEP_M=0.05` is teleop-only for now.)

**Isaac Lab + model-based RL precedent** [H/L]
- leggedrobotics/robotic_world_model (RWM, arXiv:2501.10100 / RWM-U arXiv:2504.16680) — real MBRL-on-
  Isaac-Lab, but locomotion/ANYmal, PPO-in-imagination (architectural cousin, not Dreamer) [H].
- A-SHOJAEI/dreamerv3-robotic-control — clean PyTorch DreamerV3, but DMControl/robosuite not Isaac [M].
- ⚠️ DIRECTLab/IsaacLabDreamerV3 exists but is a **bare Isaac Lab fork** — contents unverified, inspect
  before relying [L]. **DreamerV3-on-Isaac-Lab manipulation looks largely unpublished → novelty angle.**

**Single-env DreamerV3 viability** [H]
- Paper trains "one Nvidia GPU," NM512 default `envs:1` → single-env is native, very sample-efficient.
  Limits are wall-clock + VRAM, not sample count. No published wall-clock number for your exact task
  (warehouse pickup / 1000-step / RTX 5050) → **NO SOLID REFERENCE**, measure it yourself.
