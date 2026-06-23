# experiments/analyze.py
# P5 — aggregate the 18 runs and run the significance tests (spec "Uji signifikansi").
#
# Reads each run's eval_metrics.csv (written by experiments.metrics.EvalCsv), then:
#   - final success rate per (config, seed)  -> mean +/- std across the 3 seeds,
#   - sample efficiency: env steps to first reach a success-rate threshold,
#   - Mann-Whitney U p-values for the isolation comparisons in configs.ISOLATION_COMPARISONS.
#
# Mann-Whitney U is used (not a t-test) because n=3 seeds per config is too small to
# justify a normality assumption. With scipy installed we call scipy.stats.mannwhitneyu;
# otherwise we fall back to an EXACT permutation null distribution (cheap for n=3).
#
# PURE python + numpy (no Isaac/torch). Unit-tested in tests/test_experiments_analyze.py.

"""Aggregate runs (mean/std success rate, sample efficiency) + Mann-Whitney U tests."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

from experiments.configs import CONFIGS, ISOLATION_COMPARISONS, SEEDS

DEFAULT_THRESHOLD = 0.5  # success-rate threshold for the sample-efficiency metric


# ── CSV reading ────────────────────────────────────────────────────────────────

def read_run_history(csv_path: str | Path) -> list[dict]:
    """Read one run's eval_metrics.csv into a list of typed row dicts (step-sorted)."""
    rows: list[dict] = []
    with Path(csv_path).open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "step": int(float(r["step"])),
                "success_rate": float(r["success_rate"]),
                "mean_length": float(r["mean_length"]),
                "mean_return": float(r["mean_return"]),
            })
    return sorted(rows, key=lambda x: x["step"])


def final_success(rows: list[dict]) -> float:
    """Final-eval success rate = last (highest-step) row's success_rate."""
    if not rows:
        return float("nan")
    return rows[-1]["success_rate"]


def steps_to_threshold(rows: list[dict], threshold: float = DEFAULT_THRESHOLD) -> int | None:
    """Env steps at the first eval where success_rate >= threshold, else None."""
    for r in rows:
        if r["success_rate"] >= threshold:
            return r["step"]
    return None


# ── Statistics ───────────────────────────────────────────────────────────────

def aggregate(values: list[float]) -> tuple[float, float]:
    """Return (mean, population std) of a list; (nan, nan) if empty."""
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return mean, math.sqrt(var)


def _u_statistic(a: list[float], b: list[float]) -> float:
    """Mann-Whitney U for sample `a` vs `b` (ties count as 0.5)."""
    u = 0.0
    for x in a:
        for y in b:
            u += 1.0 if x > y else (0.5 if x == y else 0.0)
    return u


def _exact_two_sided_p(a: list[float], b: list[float]) -> float:
    """Exact permutation two-sided p-value for the U statistic (small n)."""
    na, nb = len(a), len(b)
    pooled = a + b
    u_obs = _u_statistic(a, b)
    u_mean = na * nb / 2.0
    target = abs(u_obs - u_mean)
    total = count = 0
    for combo in itertools.combinations(range(na + nb), na):
        ga = [pooled[i] for i in combo]
        gb = [pooled[i] for i in range(na + nb) if i not in combo]
        if abs(_u_statistic(ga, gb) - u_mean) >= target - 1e-9:
            count += 1
        total += 1
    return count / total


def mann_whitney_u(a: list[float], b: list[float]) -> tuple[float, float, str]:
    """Return (U, two-sided p, method). Uses scipy if available, else exact permutation."""
    if not a or not b:
        return float("nan"), float("nan"), "empty"
    try:
        from scipy.stats import mannwhitneyu  # type: ignore

        u, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(u), float(p), "scipy"
    except ImportError:
        return _u_statistic(a, b), _exact_two_sided_p(a, b), "exact-perm"


# ── Driver ───────────────────────────────────────────────────────────────────

def collect(results_dir: str | Path) -> dict[int, dict]:
    """Gather per-config final success rates + sample efficiency across seeds.

    Looks for <results_dir>/<config.logname>_seed<n>/eval_metrics.csv.

    Returns:
        {config_idx: {"final": [..per seed..], "steps_thr": [..], "logname": str}}
    """
    results_dir = Path(results_dir)
    out: dict[int, dict] = {}
    for cfg in CONFIGS:
        finals, steps, empty = [], [], []
        for seed in SEEDS:
            csv_path = results_dir / f"{cfg.logname}_seed{seed}" / "eval_metrics.csv"
            if not csv_path.exists():
                empty.append((seed, "missing"))
                continue
            rows = read_run_history(csv_path)
            if not rows:
                empty.append((seed, "no eval rows"))  # header-only CSV → don't silently nan
                continue
            finals.append(final_success(rows))
            s = steps_to_threshold(rows)
            if s is not None:
                steps.append(s)
        out[cfg.idx] = {"final": finals, "steps_thr": steps,
                        "logname": cfg.logname, "empty": empty}
    return out


def render_summary(data: dict[int, dict], threshold: float = DEFAULT_THRESHOLD) -> str:
    """Build the markdown summary (per-config table + isolation comparisons)."""
    lines = ["# Experiment summary\n",
             "## Final success rate (mean +/- std over seeds 0,1,2)\n",
             "| # | config | seeds | success rate | steps to "
             f"{threshold:.0%} (mean) |",
             "|---|--------|-------|--------------|------------------------|"]
    warnings: list[str] = []
    for cfg in CONFIGS:
        d = data.get(cfg.idx, {})
        finals = d.get("final", [])
        empty = d.get("empty", [])
        mean, std = aggregate(finals)
        steps = d.get("steps_thr", [])
        steps_mean = f"{sum(steps) / len(steps):,.0f}" if steps else "not reached"
        sr = f"{mean:.3f} +/- {std:.3f}" if finals else "**NO DATA**"
        if empty:
            detail = ", ".join(f"seed{s}: {why}" for s, why in empty)
            warnings.append(f"- #{cfg.idx} {cfg.logname}: "
                            f"{len(empty)}/{len(SEEDS)} run(s) without eval data ({detail})")
        lines.append(
            f"| {cfg.idx} | {cfg.logname} | {len(finals)} | {sr} | {steps_mean} |"
        )

    if warnings:
        lines += ["\n## WARNING: runs with no eval data\n",
                  "Shown as **NO DATA** / nan above -- `eval_metrics.csv` was missing or "
                  "header-only (no rows logged). The eval callback recorded nothing; check "
                  "`experiments.metrics.evaluate_policy`. Do NOT trust these rows.",
                  *warnings]

    lines += ["\n## Pairwise significance (Mann-Whitney U, two-sided)\n",
              "| comparison | what it isolates | mean A | mean B | U | p | method |",
              "|------------|------------------|--------|--------|---|---|--------|"]
    for a_idx, b_idx, what in ISOLATION_COMPARISONS:
        a = data.get(a_idx, {}).get("final", [])
        b = data.get(b_idx, {}).get("final", [])
        u, p, method = mann_whitney_u(a, b)
        ma, _ = aggregate(a)
        mb, _ = aggregate(b)
        lines.append(
            f"| #{a_idx} vs #{b_idx} | {what} | {ma:.3f} | {mb:.3f} | "
            f"{u:.1f} | {p:.4f} | {method} |"
        )
    lines.append("\n_Note: with n=3 seeds per config the smallest achievable two-sided "
                 "p is 0.1 (exact). Treat p-values as indicative; report effect sizes too._")
    return "\n".join(lines)


def main() -> None:
    """CLI: aggregate a results dir, print + write summary.md."""
    ap = argparse.ArgumentParser(description="Aggregate ablation runs + significance tests")
    ap.add_argument("--results", default="training/results/ablation",
                    help="Dir containing <logname>_seed<n>/eval_metrics.csv subdirs")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = ap.parse_args()

    data = collect(args.results)
    summary = render_summary(data, args.threshold)
    print(summary)
    out = Path(args.results) / "summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(summary, encoding="utf-8")
    print(f"\n[analyze] wrote {out}")


if __name__ == "__main__":
    main()
