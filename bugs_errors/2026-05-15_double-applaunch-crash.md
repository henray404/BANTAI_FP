# Bug: Fatal crash — double AppLauncher instantiation

**Date:** 2026-05-15  
**File:** env/warehouse_env.py + tests/test_env.py  
**Status:** [x] Fixed

---

## Error Message
```
[Fatal] [carb.crashreporter-breakpad.plugin] [crash] Thread backtrace follows:
Couldn't get crash stack trace.
terminatedByAbort = '0'
```

## What I Was Doing
Running `python tests/test_env.py`. Isaac Sim crashed fatally.

## Root Cause
`warehouse_env.py` had `AppLauncher(args_cli)` at module level.
`test_env.py` already launched AppLauncher, then imported from `warehouse_env.py`,
triggering a second AppLauncher instantiation → Isaac Sim crash.

Isaac Lab rule: AppLauncher must be instantiated EXACTLY ONCE per process,
in the entry script. Module files must NOT launch it.

## Fix
Removed AppLauncher block from `warehouse_env.py` module level.
Classes (WarehouseEnvCfg, etc.) are now importable without side effects.
AppLauncher stays only in entry scripts: `run_env.py`, `test_env.py`.

## Notes
Any file meant to be imported must NOT call AppLauncher at module level.
Only `if __name__ == "__main__"` blocks may call AppLauncher in module files.
