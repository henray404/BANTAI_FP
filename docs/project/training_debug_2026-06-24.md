# Training Debug & Fixes — 2026-06-24 (branch `jam3`)

Context summary of a debugging session bringing DreamerV3 training up on a fresh remote GPU,
then diagnosing why the agent wasn't learning and why the sim kept crashing. All fixes landed on
branch **`jam3`** (off `main`). Stage 3 training is now **stable**; the agent is training.

---

## 1. Remote environment (vast.ai)

| Item | Value |
|---|---|
| GPU | RTX 4090, **48 GB** VRAM (Ada, sm_89) — NOT the 8 GB Blackwell 5050 the old bug docs assume |
| Driver / CUDA | 575.51.03 / CUDA 12.9 |
| OS | Ubuntu container (root), `/workspace` |
| conda env | `isaaclab` (Python **3.11** — required; the container's default `main` env is 3.12 and will NOT work) |
| Isaac Sim / Lab | 5.1.0 / 0.54.4 |
| torch | 2.7.0+cu128 |

Setup that mattered:
- Render libs: `apt-get install libgl1 libglib2.0-0 libxrandr2 libxinerama1 libxcursor1 libxi6 vulkan-tools libglu1-mesa libsm6 libice6 libxt6 libxext6` (the `libGLU.so.1` / `libGLU` miss blocked the RTX renderer).
- Warehouse assets (~468 MB) are gitignored — fetched from Google Drive, unzipped to `assets/`.
- Always run **`--headless`**; keep the instance "Stop" (not "Destroy") to preserve the setup.
- Training is **resumable** from `latest.pt` (proven): re-running with the same `--logdir` continues from the last checkpoint + dataset.

---

## 2. cuDNN crash (FIXED)

**Symptom:** `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` on the first `F.conv2d`
(even a bare torch conv), internal reason `cudaGetDeviceCount(&count) != cudaSuccess`.

**Diagnosis:** CUDA/GPU/torch all healthy (conv works with `torch.backends.cudnn.enabled=False`;
`cudaGetDeviceCount` via ctypes returns 0/1). Only cuDNN's own init fails. torch 2.7.0 pins
`nvidia-cudnn-cu12==9.7.1.26`, which is **incompatible with driver 575** here.

**Fix:** upgrade cuDNN (overrides torch's pin; cuDNN 9.x is ABI-stable):
```bash
pip install --force-reinstall --no-deps -U nvidia-cudnn-cu12   # -> 9.23.2.1
```
After this, conv works with cuDNN enabled (full speed). No code change.

---

## 3. Reward hacking / success-rate = 0 (FIXED the reward side)

**Symptom:** config #4 ran to 110k but `success_rate = 0` at every eval, while `train_return`
exploded (up to **97 918**). Episodes ≥900 steps averaged return ~10 144 vs <200-step episodes ~63
— i.e. return scaled with episode *length*, not task success.

**Diagnosis (ruled out, then found):**
- NOT the per-step `deliver` reward: `success = DoneTerm(pickup_delivered)` terminates the episode
  on delivery, so deliver can't be farmed (the 98k episodes were 1000-step timeouts where
  `pickup_delivered` never fired).
- NOT CA-SLOPE: `reward/ca_slope.py` is correctly telescoping (`F = γΦ' − Φ`, terminal Φ=0), bounded.
- **Actual cause:** the PBS dense shaping (`approach`/`carry`) was reading distances that blew up
  when the robot's position diverged (physics — see §4). With a positive weight, the shaping
  faithfully reported the divergence as huge per-step reward → reward hacking.

**Fix (jam3):**
- `configs/reward_weights.yaml`: `approach`/`carry` restored **0.4** (a prior commit had set them to
  −0.01, which *removed* dense guidance → agent jittered forward/back, never grasped).
- `env/warehouse_env.py` `_update_pbs_shaping`: **clamp** `_approach_shaping`/`_carry_shaping` to ±1.0
  so a position spike can't inject thousands. Normal per-step |Δdist| ≪ 1 m, so real shaping is never
  clipped.
- `reward/ca_slope_wrapper.py` `_potential`: use **`ee_pos_world`** (env-local world) not `ee_pos`
  (base-frame delta) so `dist(ee, box_pos)` is frame-consistent (same class as the C1 fix).

---

## 4. Sim crash — PhysX CUDA error 700 (ROOT CAUSE FOUND)

**Symptom:** `omni.physx.tensors ... CUDA error: an illegal memory access` in
`GpuRigidBodyView`/`GpuArticulationView` + `PxgCudaMemoryAllocator` warning, recurring after a few
minutes (~95–234 s).

**Ruled out (with evidence):**
- **NOT GPU memory / BAR1** (unlike the 8 GB 5050 bug doc): live `nvidia-smi` showed **VRAM flat at
  ~13.9 GB / 49 GB** and **BAR1 used 7–8 MiB** right up to the crash. No leak, no exhaustion.
- **NOT base physics in general:** a **stage-3, no-curriculum run survived 8+ minutes** with no crash
  (crashes were consistently ~220 s, so 8 min clears it).
- **NOT depenetration magnitude alone:** the `max_depenetration_velocity=1.0` cap on the robot (jam3)
  cut spawn contact force from `contactN ~8e8` to `~1.7e3`, but did not stop the crash by itself.

**Root cause:** the crash only happens with **curriculum stages 1 & 2**, which **override the spawn**:
- Stage 1 = box pre-grasped (snap + weld at spawn).
- Stage 2 = base teleported to `spawn_pose_near_box(standoff=0.8 m)` next to the target box.
At `standoff=0.8 m` the chassis overlaps the rack's (explicit cuboid) collider → interpenetration →
articulation/rigid-body state corruption → CUDA error 700. Stage 3 uses the default open north
spawn → no interpenetration → **stable**.

**Mitigations applied (jam3):** robot `max_depenetration_velocity=1.0` + `max_linear_velocity=5.0`
/ `max_angular_velocity=10.0` caps in `env/warehouse_scene.py` (sound stability backstops; they
reduce the blast but do NOT fix the stage-1/2 spawn overlap).

**Still open:** stages 1/2 spawn overlaps the rack. To enable curriculum, either raise `standoff`
(0.8 → ~1.5–2.0 m, balancing grasp reachability with the frozen-arm magnetic grasp) or shrink the
rack collider cuboid. Needs in-Isaac iteration. **Until then: train stage 3 only.**

---

## 5. Other findings

- **No real category-mapping bug.** Verified the full chain `ZONE_SPECS → _zone_pos →
  _sample_targets (goal_pos=_zone_pos[c], goal_id=onehot(c)) → demo target=obs["goal"] →
  pickup_delivered`. Category c's box is routed to category c's (correct-colour) zone. A
  "wrong-category" delivery in the demo is not a code mapping bug.
- **Frozen arm is fine for the task.** Grasp is MAGNETIC: drive the base so the fixed EE is within
  ~8 cm of the box + close gripper → box auto-welds. No arm motion needed. The 8 cm window is small,
  which is exactly why the dense approach shaping (§3) matters.
- **train_ratio**: lowered 512 → **128** (`experiments/ablation.yaml`) for wall-clock (fps ~10 vs ~5).
  Committed so `git pull`/checkout won't reset it. Bump toward 256/512 for a final high-quality run.

---

## 6. Branch `jam3` — commits (newest first)

```
eb02519  chore: train_ratio 512->128 (persist)
3d00637  fix(env): cap robot max_linear/angular_velocity (error-700 backstop)
c06a765  fix(env): cap robot max_depenetration_velocity (error-700 blowup)
907785c  fix(experiments,reward): forward --curriculum in run_all; fix CA-SLOPE wrapper EE frame
e96a03a  fix(reward,env): clamp PBS shaping; restore approach/carry 0.4; step-based curriculum auto-advance
```
Files touched: `configs/reward_weights.yaml`, `experiments/ablation.yaml`, `experiments/run_all.py`,
`scripts/train_dreamer.py`, `env/warehouse_env.py`, `env/warehouse_scene.py`, `reward/ca_slope_wrapper.py`.

New mechanism: step-based curriculum auto-advance — `WarehouseGymEnv.set_stage_schedule()` +
`--curriculum "stage:frac,..."` on `train_dreamer.py`, forwarded by `run_all --curriculum`.
**Currently unusable** because stages 1/2 crash (§4) — use only after the spawn is fixed.

---

## 7. Current recommended path

Train **config #4 (DreamerV3 + CA-SLOPE), stage 3, no curriculum** — stable, with CA-SLOPE providing
the dense category-aware guidance that substitutes for curriculum:
```bash
git fetch origin && git reset --hard origin/jam3
python -m experiments.run_all --device 0 --only 4 --seeds 0 --steps 150000 \
  --results training/results/c4stage3 2>&1 | tee c4stage3.log
```
Health signals (NOT episode length, which is short early and normal): `train_return` trending up,
occasional ~+8 (grasp), `model_loss`/`image_loss` decreasing. It is sparse + hard; success > 0 needs
substantial training. If crashes ever recur on stage 3, wrap in an auto-resume loop (resumable).

## 8. TODO / open
- [ ] Fix stage-1/2 spawn overlap to re-enable curriculum (standoff / rack collider). Needs Isaac.
- [ ] Spawn-in-collision: some episodes still terminate `crashed` at `ep_len=0` in the receiving area.
- [ ] Confirm learning: success_rate still 0 — verify it rises with stage-3 + CA-SLOPE + more steps.
- [ ] Optional: bump train_ratio back up for a final quality run if wall-clock allows.
