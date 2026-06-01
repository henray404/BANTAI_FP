# Bug: SDP Camera Crash on RTX 5050 Blackwell

**Date:** 2026-05-22
**Status:** FIXED (replaced CameraCfg with TiledCameraCfg)

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
