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
