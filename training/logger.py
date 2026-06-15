# training/logger.py
# Person 5 — experiment logging wrapper (Weights & Biases).
#
# Import-guarded: if wandb is not installed, every call is a safe no-op that also
# prints scalars to stdout. This lets the whole pipeline run on a clean env
# without wandb (e.g. CI / the pinned isaaclab env) and light up logging once
# `pip install wandb` + `wandb login` are done.

"""Thin W&B logging wrapper with a stdout no-op fallback."""

from __future__ import annotations

from typing import Any

try:
    import wandb  # type: ignore

    _HAS_WANDB = True
except ImportError:  # wandb optional
    wandb = None
    _HAS_WANDB = False


class Logger:
    """W&B run wrapper.

    Usage:
        log = Logger(project="bantai-warehouse", config=cfg, name="sac_seed0")
        log.log({"reward/mean": r}, step=global_step)
        log.finish()
    """

    def __init__(
        self,
        project: str = "bantai-warehouse",
        config: dict[str, Any] | None = None,
        name: str | None = None,
        mode: str = "online",
        print_every: int = 1000,
    ):
        """Start a W&B run if available; otherwise fall back to stdout."""
        self.enabled = _HAS_WANDB and mode != "disabled"
        self._print_every = print_every
        self._last_print = 0
        if self.enabled:
            self._run = wandb.init(project=project, config=config, name=name, mode=mode)
        else:
            self._run = None
            if not _HAS_WANDB:
                print("[logger] wandb not installed — logging to stdout only "
                      "(pip install wandb to enable).")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a dict of scalars. Throttled stdout echo when wandb is absent."""
        if self.enabled:
            self._run.log(metrics, step=step)
            return
        if step is None or step - self._last_print >= self._print_every:
            self._last_print = step or 0
            flat = " ".join(f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                            for k, v in metrics.items())
            print(f"[step {step}] {flat}")

    def watch(self, model) -> None:
        """Track model gradients/params (no-op without wandb)."""
        if self.enabled:
            self._run.watch(model)

    def finish(self) -> None:
        """Close the run."""
        if self.enabled:
            self._run.finish()
