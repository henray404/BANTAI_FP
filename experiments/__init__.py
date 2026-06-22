"""Experiment harness: 2x2 factorial ablation (CA-SLOPE x Visual HER) + 2 baselines.

Six configurations x 3 seeds = 18 runs. See experiments/configs.py for the registry
and docs/experiments/README.md for the protocol.

Also hosts the headless toy eval + trajectory recording harness (eval_harness.py, run_eval.py,
scenarios.py, toy_pickup_env.py, scripted_policy.py) — runs scenarios, logs per-step traces +
per-episode summary, and optionally records replayable runs for per-scenario ranking.
"""
