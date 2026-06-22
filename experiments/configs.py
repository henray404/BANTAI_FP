# experiments/configs.py
# P5 — single source of truth for the ablation study.
#
# Design (CA-SLOPE-only study, trimmed 2026-06-23 from the original 6-config 2x2):
#   CA-SLOPE on/off on DreamerV3 (#3 vanilla vs #4 ca_slope) + one PPO baseline (#2) = 3 configs.
#   Visual HER (#5, #6) and the SAC baseline (#1) dropped — focus on CA-SLOPE.
#   Each config x 3 seeds {0,1,2} = 9 runs, 200_000 env steps each,
#   periodic eval every 10_000 steps over 5 episodes.
#   (idx kept at 2/3/4 from the original table so lognames + prior results stay stable.)
#
# This module is PURE python (no Isaac / torch import) so it is importable from the
# orchestrator (run_all.py), the analysis script (analyze.py), and the entry scripts
# (train_dreamer.py / train_sac.py) without launching the simulator.

"""Registry of the experiment configurations + per-config hyperparameters + papers."""

from __future__ import annotations

from dataclasses import dataclass, field

# Shared experiment budget (spec "Anggaran pelatihan dan seed").
TOTAL_STEPS: int = 200_000
SEEDS: tuple[int, ...] = (0, 1, 2)
EVAL_EVERY: int = 10_000
EVAL_EPISODES: int = 5


@dataclass(frozen=True)
class ExperimentConfig:
    """One experiment configuration in the ablation matrix.

    Attributes:
        idx:         Configuration number 1..6 from the spec table.
        name:        Short slug used for logdirs and CSV filenames.
        algo:        Training stack: "sac" | "ppo" | "dreamer".
        ca_slope:    Enable Category-Aware SLOPE potential-based reward shaping.
        visual_her:  Enable Visual HER episode relabeling.
        kind:        Human label ("model-free baseline" / "model-based baseline" /
                     "ablation" / "full (proposed)").
        isolates:    What this config isolates per the spec's isolation logic.
        papers:      Reference keys (see docs/research/referensi.md) motivating it.
        overrides:   Extra hyperparameter overrides merged on top of the algo default.
    """

    idx: int
    name: str
    algo: str
    ca_slope: bool
    visual_her: bool
    kind: str
    isolates: str
    papers: tuple[str, ...]
    overrides: dict = field(default_factory=dict)

    @property
    def logname(self) -> str:
        """Logdir/run slug: e.g. 'c3_dreamer_vanilla'."""
        return f"c{self.idx}_{self.name}"


# ── The 6 configurations ──────────────────────────────────────────────────────
# papers reference the numbered entries in docs/research/referensi.md.
CONFIGS: tuple[ExperimentConfig, ...] = (
    ExperimentConfig(
        idx=2, name="ppo", algo="ppo", ca_slope=False, visual_her=False,
        kind="model-free baseline",
        isolates="model-free floor (on-policy)",
        papers=("PPO-Schulman2017-#19", "SB3-DLR-RM"),
    ),
    ExperimentConfig(
        idx=3, name="dreamer_vanilla", algo="dreamer", ca_slope=False, visual_her=False,
        kind="model-based baseline",
        isolates="pure world-model effect (vs #2)",
        papers=("DreamerV3-Hafner2023-#1", "DayDreamer-#4", "NM512-torch"),
    ),
    ExperimentConfig(
        idx=4, name="dreamer_caslope", algo="dreamer", ca_slope=True, visual_her=False,
        kind="ablation (proposed)",
        isolates="CA-SLOPE contribution over the world model (vs #3)",
        papers=("DreamerV3-Hafner2023-#1", "PBRS-Ng1999-#15", "Devlin2012-#16"),
    ),
)

# Pairwise comparisons to report (spec "Uji signifikansi"): (A_idx, B_idx, what).
# HER configs (#5, #6) and the SAC baseline (#1) dropped 2026-06-23 — CA-SLOPE-only study.
ISOLATION_COMPARISONS: tuple[tuple[int, int, str], ...] = (
    (4, 3, "CA-SLOPE over world model"),
    (3, 2, "model-based vs PPO"),
)


def by_idx(idx: int) -> ExperimentConfig:
    """Return the configuration with the given 1-based index."""
    for c in CONFIGS:
        if c.idx == idx:
            return c
    raise KeyError(f"No experiment config with idx={idx} (valid: 2, 3, 4).")


def by_name(name: str) -> ExperimentConfig:
    """Return the configuration whose name or logname matches `name`."""
    for c in CONFIGS:
        if name in (c.name, c.logname):
            return c
    raise KeyError(f"No experiment config named {name!r}.")


def all_runs() -> list[tuple[ExperimentConfig, int]]:
    """Return every (config, seed) pair — the full 18-run schedule, in run order."""
    return [(c, s) for c in CONFIGS for s in SEEDS]


if __name__ == "__main__":
    # Smoke check: 3 configs (CA-SLOPE-only study), 9 runs, comparisons reference real indices.
    assert len(CONFIGS) == 3, len(CONFIGS)
    assert len({c.idx for c in CONFIGS}) == 3
    assert len(all_runs()) == 9
    for a, b, _ in ISOLATION_COMPARISONS:
        by_idx(a); by_idx(b)
    for c in CONFIGS:
        print(f"#{c.idx} {c.logname:24s} algo={c.algo:8s} "
              f"ca_slope={c.ca_slope!s:5s} her={c.visual_her!s:5s} | {c.kind}")
    print("OK: 3 configs, 9 runs.")
