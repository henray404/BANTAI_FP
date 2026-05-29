# Bug: h5py DLL load failed

**Date:** 2026-05-14  
**File:** Isaac Lab tutorial — create_cartpole_base_env.py  
**Status:** [x] Fixed

---

## Error Message
```
ImportError: DLL load failed while importing _errors
import h5py
Windows fatal exception: code 0xc0000139
```

## What I Was Doing
Running `create_cartpole_base_env.py --num_envs 4` for the first time.

## Root Cause
h5py version installed was incompatible with Isaac Lab 5.x on Windows.

## Fix
```bash
pip uninstall h5py -y
pip install h5py==3.11.0
```

## Notes
Always check h5py version after fresh Isaac Lab install on Windows.
