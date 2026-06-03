# Bug: SDP Camera Crash on RTX 5050 Blackwell

**Date:** 2026-05-22
**Status:** ✅ RESOLVED 2026-06-03 — NVIDIA driver downgrade 591.84 → 580.88 (see resolution at bottom).
  `tests/test_env.py --num_envs 1` (camera ON) now reports **ALL PASS** — first time ever.

## Symptom

Access violation crash at ~31 seconds on every run (headless or windowed):
```
exception: access violation
lastCommand = CreateAttrCommand(attr_name=state:_sdp_intergraph_downstream_node_handles_,...)
UptimeSeconds = 31
```

Scene loads successfully (walls + zones get BindMaterialCommand). Crash always in
camera SDP (Synthetic Data Pipeline) OmniGraph initialization.

## Environment

- GPU: NVIDIA GeForce RTX 5050 Laptop GPU (Blackwell architecture)
- Driver: 591.84
- Isaac Sim: 5.1.0
- Isaac Lab: 5.x

## Root Cause

CameraCfg uses omni.replicator / SDP OmniGraph pipeline for synthetic data generation.
This pipeline has an access violation when initializing
state:_sdp_intergraph_downstream_node_handles_ on Blackwell GPU + Isaac Sim 5.1.

Steps confirmed NOT to fix it:
1. Closing background GPU processes
2. Running with --headless flag
3. Changing camera prim path (Robot/chassis/onboard_cam -> Robot/onboard_cam)

## Fix

Replace CameraCfg with TiledCameraCfg in warehouse_scene.py.
TiledCamera renders via CUDA/warp kernel, bypasses SDP OmniGraph entirely.

Sensor data format difference:
- CameraCfg:      data.output["rgb"] -> Tensor(B, H, W, 4) uint8 RGBA
- TiledCameraCfg: data.output["rgb"] -> Tensor(B, H, W, 3) float32 [0, 1] RGB

camera_rgb() in warehouse_env.py handles both via dtype check.

## Update 2026-06-01: TiledCamera does NOT bypass SDP on Isaac Sim 5.1

The "bypasses SDP OmniGraph entirely" claim above is FALSE on Isaac Sim 5.1.
`explore_scene.py` crashed with the identical signature
(`state:_sdp_intergraph_downstream_node_handles_`, access violation, ~36 s uptime).

Log evidence: crash fires on the line immediately after
`XFormPrimView over '.../Robot/onboard_cam'` — i.e. during TiledCamera init, not the
human viewport. (`--headless` did not help before, confirming the sensor camera — not the
viewport — is the trigger.) BAR1 VRAM was near-exhausted (8164/8192 MiB), which aggravates
the SDP allocation crash on the 8 GB RTX 5050.

### Workaround for explore_scene.py (visual inspection only)
That script never reads camera tensors, so it strips the sensors before scene build:
```python
scene_cfg.camera = None
scene_cfg.contact_sensor = None
```
No camera prim -> no SDP graph -> no crash. (`enable_cameras` left off.)

### OPEN: RL env (warehouse_env.py / run_env.py) still affected
The RL env REQUIRES the camera for `obs["pixels"]`, so it cannot strip it. It will hit this
same SDP crash on Blackwell. Real fix still needed — candidates to try:
- newer NVIDIA driver / Isaac Sim patch for Blackwell SDP
- reduce VRAM pressure (fewer envs, smaller tiled-camera resolution already 64x64)
- alternative RGB capture path that avoids the replicator/SDP annotator graph

---

## Update 2026-06-03: DEEP RESEARCH — driver is the prime suspect [High]

Web research across NVIDIA Developer Forums + IsaacLab GitHub. The crash family
(`rtx.scenedb.plugin` / SDP / replicator access-violation on Blackwell sm_120) is **widely
reported and driver-linked**, not specific to our scene.

### Root cause (most likely): NVIDIA driver branch
- **Our driver = 591.84** (confirmed via `nvidia-smi`). RTX 5050 Laptop, 8 GB.
- **NVIDIA staff (VickNV):** *"known issue with the 595.xx driver branch on Blackwell GPUs"* →
  downgrade. Multiple Blackwell cards (5060 Ti, 5070 Ti, 5090) crash on 595.79, **fixed by
  downgrade to 591.74**.
- **Windows-validated driver for Isaac Sim 5.1 = 580.88** (Isaac Lab install docs).
- Isaac Sim **fails to detect CUDA on 595.79, works on 580** (IsaacSim issue #537).
- → Our 591.84 sits ABOVE the validated 580.88 and the known-good 591.74. Same rendering-pipeline
  regression family. **It was never the camera config that was fundamentally broken — likely the
  driver.** (Our earlier "CameraCfg not supported on Blackwell" conclusion was reached UNDER this
  bad driver and may be false.)

### Camera backend: TiledCamera vs standard Camera on Blackwell
- IsaacLab **issue #4951**: on RTX 5090 Blackwell + Isaac 5.1, **TiledCamera hangs** (100% CPU,
  `omni.replicator` tiled pipeline) — and the **workaround is standard `Camera`/`CameraCfg`**
  (`rgb = camera.data.output["rgb"].squeeze()[..., :3]`), works for `num_envs=1`.
- This is the OPPOSITE of our 2026-05-22 conclusion (we moved CameraCfg→TiledCamera). On the wrong
  driver BOTH fail; the right test is: fix driver first, then A/B the two backends.

### ACTION PLAN (ordered, cheapest first)
1. **Downgrade NVIDIA driver to 580.88** (Windows-validated for Isaac Sim 5.1). Fallback known-good:
   591.74. Clean install (DDU recommended). **This is the #1 fix — try before any code change.**
2. After downgrade: `python tests/test_env.py --num_envs 1` with camera ON → expect PASS.
3. If TiledCamera still hangs/crashes: A/B test **standard `Camera`/`CameraCfg`** (#4951 workaround).
   Keep `num_envs=1`.
4. If still failing: reduce VRAM pressure (BAR1 was 8164/8192 MiB — near-full). For pure-nav
   camera bring-up, temporarily disable the 54 decoration boxes + props to free VRAM for the SDP
   allocation.
5. **Plan B (worst case):** render/train on a non-Blackwell GPU or cloud (Isaac Sim 5.1 headless).
   Keep RTX 5050 local for camera-strip dev/teleop only.

### Notes
- Forum 365335 (driver downgrade) is a GUI/preferences-window crash where **headless works**. OUR
  crash is the camera SDP sensor graph, which `--headless` does NOT fix (confirmed in our logs).
  Different symptom, same driver-branch root cause.
- Validated config target: **Isaac Sim 5.1 + driver 580.88 + Windows 11**.

### Sources
- IsaacLab #4951 — TiledCamera hangs on RTX 5090 Blackwell: https://github.com/isaac-sim/IsaacLab/issues/4951
- Forum — driver downgrade 595.79→591.74 fixes Blackwell crash: https://forums.developer.nvidia.com/t/isaac-sim-5-1-gui-crash-access-violation-on-rtx-5070-ti-blackwell-fixed-by-driver-downgrade-to-591-74/365335
- Forum — RTX 5060 Ti rtx.scenedb.plugin crash, NVIDIA "595.xx branch known issue": https://forums.developer.nvidia.com/t/isaac-sim-5-1-crashes-on-startup-with-rtx-5060-ti-blackwell-sm-120-rtx-scenedb-plugin-crash/366252
- IsaacSim #537 — no CUDA on 595.79, works on 580: https://github.com/isaac-sim/IsaacSim/issues/537
- Isaac Lab install requirements (validated driver): https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

---

## ✅ RESOLUTION 2026-06-03 — driver was the root cause, confirmed [High]

Deep-research hypothesis confirmed by experiment. **Driver downgrade 591.84 → 580.88** (Windows-
validated for Isaac Sim 5.1, DDU clean install) eliminated the SDP camera crash entirely. The crash
signature `state:_sdp_intergraph_downstream_node_handles_` (access violation, ~31–36 s uptime)
**no longer occurs**. The 2026-05-22 "CameraCfg→TiledCamera" conclusion was a red herring reached
under the bad driver — the camera config was never the fundamental problem.

`nvidia-smi` confirms `Driver Version: 580.88` on the RTX 5050 Laptop GPU.

### Two follow-on code bugs surfaced once the SDP crash was gone (both fixed):

1. **GPU contact-view corruption → CUDA illegal memory access on `env.reset()`.**
   `warehouse_scene.py` `ContactSensorCfg.filter_prim_paths_expr=["…/Rack_.*", "…/wall_.*"]`.
   A PhysX filter expr must resolve to **exactly 1 prim per env**; `Rack_.*` matched 18 and
   `wall_.*` matched 4 → `omni.physx.tensors: Filter pattern … expected 1, found 18` → corrupted
   GPU articulation/rigid-body view → `CUDA error: an illegal memory access` flood (exit 9).
   **Fix:** removed `filter_prim_paths_expr` entirely. `collision_penalty` reads
   `net_forces_w_history` (net force on the chassis), never the per-object force matrix, so the
   filter was dead config. Net-force sensor needs no filter.

2. **Reward broadcast error `[1]` vs `[1,1]` in `env.step()`.**
   `reward_manager.py:152  self._reward_buf += value` raised
   `RuntimeError: output with shape [1] doesn't match the broadcast shape [1, 1]`.
   `collision_penalty` returned `net_forces_w_history[:, 0, :].norm(dim=-1)` = `(N, B)` = `[1,1]`
   (the body dim B leaked) instead of `(N,)`.
   **Fix:** `…[:, 0, :, :].norm(dim=-1).amax(dim=-1)` → max contact magnitude over bodies → `(N,)`.

### Verification (camera ON, num_envs=1)
`python tests/test_env.py --num_envs 1` → **ALL PASS** (10/10):
instantiate, reset() dict, pixels (1,3,64,64), position (1,3), goal (1,3), goal_emb (1,512),
action_space Box(2,), step()×10, reward (1,), close().

### Also confirmed this session (smoke_test.py)
- Robot loads: 12 joints, 19 bodies. `base_link` **exists** → contact-sensor prim_path verified.
- Base joints verified: `dummy_base_prismatic_x/y_joint`, `dummy_base_revolute_z_joint`.

### Still OPEN (separate, non-blocking)
- **smoke_test.py base-motion verdict is INCONCLUSIVE** (disp ≈ 0.000 m). Likely a *measurement*
  bug: `_base_xy()` reads `robot.data.root_pos_w`, but the articulation root is the fixed `world`
  link, so root pose never moves while `base_link` translates via the prismatic joints. Fix the
  smoke test to read the `base_link` body pose (`body_pos_w[base_link_idx]`) before trusting the
  world-vs-body-frame verdict. Base may actually be driving fine.
- **Base spawn height z = 0.609 m** (expected ~0.0–0.3) — verify it's the `base_link` frame origin,
  not the chassis floating.
- **Recurring zombie:** `simulation_app.close()` hangs on Blackwell after a clean run; each finished
  process keeps spinning a CPU core + holding GPU mem until killed manually. Kill stale `python.exe`
  between runs.
