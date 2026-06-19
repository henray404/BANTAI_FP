# TRAINING on a 2-GPU PC (2× RTX 2080 Ti) — DreamerV3 Warehouse Pickup

Run DreamerV3 training on a desktop with **two RTX 2080 Ti** (Turing, 11 GB each).
This guide is the *delta* on top of the base install — it does **not** repeat Isaac Lab
setup. Do the base install first, then come back here.

> **Base install:** `docs/setup/INSTALL.md` (steps 1–8). Follow it, with the
> Turing exceptions in §1 below.
> **Why a different PC:** the reference RTX 5050 is Blackwell and crashes silently
> during render (camera SDP). Turing (2080 Ti) has none of that — see
> `bugs_errors/2026-06-19_dreamerv3-vanilla-run-blackwell-recur.md`.

---

## 0. What you need (checklist)

| Need | Detail |
|---|---|
| 2× RTX 2080 Ti | Turing sm_75, 11 GB each. One **process per GPU** (VRAM is **not** pooled). |
| NVIDIA driver | any recent **CUDA 12.8-capable** Studio/Game-Ready driver. **No 580.88 pin** (that pin is a Blackwell-only workaround). `nvidia-smi` → `CUDA Version: 12.8`+. |
| System RAM | ≥ 32 GB recommended — each Isaac Sim process eats ~6–8 GB RAM. Two parallel runs = two sims. |
| Disk | ~15 GB (Isaac Sim ~10 GB + assets ~0.5 GB + repo). |
| OS | Windows 11 assumed (matches INSTALL.md). Linux: swap `.bat`→`.sh`, and `$env:VAR=..` → `export VAR=..`. |
| Repo branch | `feat/env-pickup-migration` **with the DreamerV3 fix committed** (see §-1). |

---

## -1. FIRST, on the CURRENT (5050) PC: push the fix

The DreamerV3 pickup-migration fix (obs adapter + action 6-dim + config) may be
**uncommitted**. If so, the 2-GPU PC will clone the *old broken* adapter. Commit + push
before cloning:

```powershell
git status                      # see the uncommitted edits
git add models/dreamerv3/ tests/test_obs_adapter.py bugs_errors/
git commit -m "fix(dreamerv3): migrate vanilla NM512 path to pickup obs/action contract"
git push origin feat/env-pickup-migration
```

Verify on GitHub the branch shows the new commit before continuing on the other PC.

---

## 1. Base Isaac Lab install (with Turing exceptions)

Do `docs/setup/INSTALL.md` steps 1–8 on the 2-GPU PC, **except**:

- **Step 1.1 (driver):** any CUDA 12.8-capable driver. Skip the Blackwell 580.88 pin talk.
- **Step 5 (PyTorch):** install the **same** `torch==2.7.0 torchvision==0.22.0 --index-url .../cu128`.
  The cu128 wheels include Turing (sm_75) kernels — they work on 2080 Ti. Do **not** downgrade CUDA.
- Everything else (conda env `isaaclab`, Isaac Sim 5.1, `isaaclab.bat --install`, assets copy) is identical.

Verify torch sees **both** GPUs:
```powershell
python -c "import torch; print(torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```
Expected: `2 ['NVIDIA GeForce RTX 2080 Ti', 'NVIDIA GeForce RTX 2080 Ti']`.

---

## 2. Install the ML / DreamerV3 deps

After the base env works, add the training stack (INTO the `isaaclab` env):

```powershell
conda activate isaaclab
pip install -r requirements-ml.txt
```

This pulls the vendored NM512 DreamerV3 deps (`gym==0.22`, `ruamel.yaml`, `einops`,
`moviepy`) + baselines (`stable-baselines3`, `tensorboard`, `wandb`).

**Guard the torch pin** — some deps may try to move torch. Verify it survived:
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# MUST still print: 2.7.0+cu128 True
```
If torch changed, reinstall it (INSTALL.md step 5) with `--no-deps`.

---

## 3. Get the project + assets

```powershell
cd C:\Users\<you>\Documents   # wherever
git clone https://github.com/henray404/BANTAI_FP.git
cd BANTAI_FP
git checkout feat/env-pickup-migration     # the branch with the fix
```

Copy the NVIDIA warehouse assets — **INSTALL.md step 8** (assets are gitignored, ~0.5 GB).

---

## 4. Verify before training

```powershell
# Pure CPU — no Isaac
pytest tests/test_layout_grid.py tests/test_obs_adapter.py -v

# DreamerV3 config + wrapper build (no GPU)
python -c "import sys; sys.path.insert(0,'.'); from models.dreamerv3.config import build_config; c=build_config(); print('task',c.task,'mlp',c.encoder['mlp_keys'])"

# Full env contract (needs Isaac, 1 GPU)
python tests/test_env.py --num_envs 1
```
All green → ready to train.

---

## 5. Run DreamerV3 — single GPU (start here)

Always do **one** run first to confirm the Turing PC clears the point the 5050 died at.

```powershell
conda activate isaaclab
python scripts/train_dreamer.py --headless --steps 200000 --logdir training/results/dreamerv3_seed0 *> training/results/run_seed0.log
```

Watch the console (or `Get-Content training/results/run_seed0.log -Tail 5 -Wait`) for this sequence:
```
Prefill dataset (2000 steps).
[0] dataset_size ... train_return ...
Logger: (2000 steps).
Simulate agent.
[1000] ... / train_return ... / fps ...   ← THIS line = training loop alive. 5050 never reached it.
```
If `[1000] ...` appears, the Blackwell blocker is gone and real training is running.

---

## 6. Run BOTH GPUs — two seeds in parallel

VRAM is **not** pooled. Use the second card for a **second run**, not a bigger one.
This halves wall-clock for the paper's "N seeds" requirement. No code change — steer each
process to one GPU with `CUDA_VISIBLE_DEVICES` (each process then sees its card as `cuda:0`).

**Terminal A:**
```powershell
conda activate isaaclab
$env:CUDA_VISIBLE_DEVICES=0
python scripts/train_dreamer.py --headless --seed 0 --steps 200000 --logdir training/results/dreamerv3_seed0 *> training/results/run_seed0.log
```

**Terminal B (new window):**
```powershell
conda activate isaaclab
$env:CUDA_VISIBLE_DEVICES=1
python scripts/train_dreamer.py --headless --seed 1 --steps 200000 --logdir training/results/dreamerv3_seed1 *> training/results/run_seed1.log
```

> ⚠️ Use **different `--logdir` per run** (shown above) — same logdir = clobbered metrics.
> ⚠️ Start run A, let its sim finish booting (~1 min), THEN start run B — two sims booting at once can spike RAM.

Verify both cards are working:
```powershell
nvidia-smi   # both GPU 0 and GPU 1 should show a python process + memory in use
```

---

## 7. Read the results — is it any good?

Logs land in each `--logdir`: console, `metrics.jsonl`, and TensorBoard event files.

```powershell
tensorboard --logdir training/results        # serves both runs at http://localhost:6006
```

Metrics that matter (per run):

| Metric | Meaning | Good sign |
|---|---|---|
| `train_return` / `eval_return` | episode reward sum | **trends up** over steps |
| world-model `*_loss`, `kl` | image-recon / reward / KL | decreasing or stable, **not NaN** |
| `fps` | sim speed | sanity check |

**Reading "good" from return** (terminal bonuses are big and discrete):
```
grasp_success    = +5   (one-time)
delivery_success = +10  (one-time)
dense terms      = small (±0.01·dist, -0.005/step)
```
- return stuck near 0 / negative → never grasping. **bad.**
- return climbs toward ~+5 → learning to **grasp**.
- return ~+10..+15 → grasp **+ deliver**. **good.**

Reference: a random policy episode logs `train_return ≈ -13.5` during prefill. Beating
that and rising = learning.

> **Gap:** success rate is **not** logged explicitly (env `step` info is empty). For a
> paper "% delivery success" curve, env must expose `grasp_success`/`delivery_success` in
> `info` and NM512 must log them as scalars — not wired yet.

---

## 8. Troubleshooting (Turing PC)

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` False | wrong torch build | reinstall `torch==2.7.0+cu128` (INSTALL.md §5) |
| `CUDA out of memory` traceback during training | batch too big for 11 GB (unlikely) | lower `batch_size`/`batch_length` in `models/dreamerv3/config.py` `WAREHOUSE_OVERRIDES` |
| Both runs slow / RAM thrash | two Isaac sims at once | stagger their start; ensure ≥32 GB RAM; or run seeds sequentially |
| Run B uses GPU 0 too | `CUDA_VISIBLE_DEVICES` not set in that shell | set it **before** launching python, in the **same** terminal |
| Racks/boxes render pink | external MDL materials not copied | expected, non-fatal (INSTALL.md §8 note) |
| Process exits, **no Python traceback** | native/GPU crash | should NOT happen on Turing; if it does, capture `run_seed*.log` and check the last 30 lines |
| Zombie `python.exe` after a run | Isaac `close()` hang | `Stop-Process -Name python -Force` between runs (kill TensorBoard separately) |

---

## Reference
- Base install: `docs/setup/INSTALL.md`
- Why move off Blackwell: `bugs_errors/2026-06-19_dreamerv3-vanilla-run-blackwell-recur.md`
- Project map: `CLAUDE.md` · Env design: `docs/specs/environment.md`
- DreamerV3 config: `models/dreamerv3/config.py` · vendored NM512: `models/dreamerv3/vendor/`
