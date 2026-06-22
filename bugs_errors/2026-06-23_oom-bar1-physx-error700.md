# Bug: Training crashes ~137s in — GPU BAR1 exhaustion → PhysX CUDA error 700 (8GB Blackwell)

**Date:** 2026-06-23
**File:** scripts/train_dreamer.py (full warehouse scene on 8GB RTX 5050)
**Status:** [ ] Fixed  [x] Mitigated (scene-size knobs + run headless) — verify in sim

---

## Error (key lines, full log truncated)
```
[omni.physx] PhysX error: GPU integrateCoreParallel fail to launch kernel!!
[omni.physx] PhysX error: SynchronizeStreams cuEventRecord failed with error 700
[omni.physx.tensors] CUDA error: an illegal memory access was encountered ... Failed to fetch DOF velocity
Exception: Failed to get DOF velocities from backend
[carb.cudainterop] CUDA error 700: cudaErrorIllegalAddress
[carb.cudainterop] Failed to import external memory in CUDA
[gpu.foundation] Cannot create cuda external memory for resource!
```
Crash metadata:
- `commandLine = '... train_dreamer.py --stage 2'`  → **NO `--headless`** (a viewport window rendered on top of the camera).
- `BAR1 Memory Usage: Total 8192 MiB, Used 8164 MiB, Free 28 MiB`  → **PCIe aperture FULL.**
- `gpu_0 = RTX 5050 Laptop (8GB)`, driver 580.88 (the pinned-good one — NOT the old SDP-crash driver).
- `lastCommand = CreateAttrCommand(... _sdp_intergraph_downstream_node_handles_)` (camera SDP render graph).
- `lastCommands` = BindMaterialCommand on box meshes (fragile_3/regular_3/heavy_3/4/5 …).
- Uptime 138 s before the crash (ran, then died — not an init failure).

## Root cause
GPU memory / BAR1 aperture exhaustion on the 8 GB Blackwell. The full scene (18 racks + 54 shelf
decks + **18 dynamic boxes** + 8 props + materials) **+ the 64×64 camera render + a viewport window
(not headless)** fills the 8192 MiB BAR1 aperture. Once it's full, CUDA external-memory import fails
→ render/PhysX buffers can't be created → the PhysX GPU solver dereferences invalid memory →
`error 700` (illegal address) cascades through articulation/rigid-body views. The `_sdp_` last
command points at the camera render graph as the immediate trigger under memory pressure.

NOT a clean RAM OOM (system RAM had ~3.6 GB free); it is GPU-side. NOT the old Blackwell SDP camera
crash (that was driver 591.x; this is the pinned 580.88).

## Mitigation
1. **Run headless** — `python scripts/train_dreamer.py --stage 2 --headless`. Removes the viewport
   render (a second full render path); biggest single win. The camera obs still renders.
2. **Spawn fewer boxes / skip props** — new `scene:` knobs in `configs/env_config.yaml`
   (read by `env/warehouse_scene._scene_knobs`):
   - `num_boxes: 3` → spawn only 1 box per category instead of 18 (category-balanced subset;
     target sampled from the spawned set). Far fewer rigid bodies + material binds + GPU contact pairs.
   - `spawn_props: false` → skip the 8 decorative props.
   `num_boxes: 0` (default) = all 18 = unchanged for full-scene / other consumers.

NOTE: hiding a box (visibility off) does NOT free VRAM — the rigid body + geometry stay loaded. Only
NOT spawning it saves memory. Racks + camera still dominate VRAM; 8 GB is marginal for the full scene
— the robust path remains the offline split (collect light locally → train DreamerV3 on A100).

## Related
- bugs_errors/2026-05-22_sdp-camera-crash-blackwell.md (different: driver 591.x SDP crash)
- docs/project/training_readiness_2026-06-22.md
