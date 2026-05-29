# Bug: RigidObjectCfg + UsdFileCfg Fails for Cardboard Box USDs

**Date:** 2026-05-22
**Status:** FIXED (items reverted to CuboidCfg primitives)

## Error

```
RuntimeError: Failed to find a rigid body when resolving '/World/envs/env_.*/fragile_0'.
Please ensure that the prim has 'USD RigidBodyAPI' applied.
  File: isaaclab/assets/rigid_object/rigid_object.py:490 _initialize_impl
```

## Cause

NVIDIA DT cardboard box USDs (CubeBox_A0x_XXcm_PR_NVD_01.usd) have a nested mesh
hierarchy inside the USDC binary. Isaac Lab's UsdFileCfg applies RigidBodyAPI to the
root prim, but the root prim of these USDs is an Xform with no physics anchor.
_initialize_impl cannot find a RigidBodyAPI prim after spawning.

## Fix

Items reverted to CuboidCfg (primitive shapes) with:
- Brown cardboard color: (0.72, 0.53, 0.30)
- Different sizes for category encoding (size distinction preserved for CLIP):
  - fragile: 0.21 x 0.21 x 0.21 m
  - regular: 0.32 x 0.32 x 0.32 m
  - heavy:   0.52 x 0.52 x 0.52 m

CuboidCfg properly applies RigidBodyAPI to the spawned prim root.

## Note on Rack MDL Errors

Rack_A01 USD references shared MDL materials (../../../../../Materials/Base/Metals/...).
Those reference the Warehouse_NVD SubUSDs folder (1.2 GB, not copied).
Rack geometry loads correctly, materials appear as pink/default. Non-fatal.
