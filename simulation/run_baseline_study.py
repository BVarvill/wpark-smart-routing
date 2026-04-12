"""
run_baseline_study.py — Statistical comparison of Nearest-Entrance vs
Floor-Directed across many simulated days.

Runs the cellular simulator N times per policy (default 1000), each with
a different random seed, and reports:
  • mean ± 95% confidence interval for every metric
  • a paired t-test on the headline "time saved per car" delta
  • CSV of per-run numbers (for further analysis)

Usage:
    cd "/Users/benvarvill/Downloads/WPark /simulation"
    python run_baseline_study.py                   # 1000 runs, peak=60
    python run_baseline_study.py --runs 200        # faster
    python run_baseline_study.py --peak 80         # higher demand
    python run_baseline_study.py --out results.csv
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import statistics
import sys
import time
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from carpark import build_demo_carpark
from demand import build_synthetic_demand
from engine import SimulationEngine


POLICIES = ("nearest_entrance", "floor_directed", "greedy_smart", "neural_smart")
METRICS_TO_TRACK = (
    "avg_cruise_time",           # seconds
    "avg_walk_time",             # seconds
    "avg_total_wasted",          # seconds
    "avg_queue_wait_seconds",    # seconds
    "correct_floor_pct",         # %
    "vehicles_served",           # count
    "vehicles_rejected",         # count
    "total_vehicles",            # count
    "total_extra_spend_daily",   # £ — Option-B revenue metric
    "avg_extra_spend_per_car",   # £
)


def _one_run(policy: str, peak_rate: int, day_type: str, seed: int) -> Dict[str, float]:
    """Run a single simulation and return a dict of metric values."""
    cp = build_demo_carpark()
    demand = build_synthetic_demand("Demo", peak_arrivals_per_hour=peak_rate)
    eng = SimulationEngine(cp, demand, policy, day_type, random_seed=seed)
    eng.generate_arrivals()
    m = eng.run()
    return {k: float(getattr(m, k)) for k in METRICS_TO_TRACK}


def _mean_ci(values: List[float], confidence: float = 0.95):
    """Return (mean, half_width_of_CI).  Uses normal-approximation CI
    which is fine for N>=30 sample sizes."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = statistics.fmean(values)
    if n == 1:
        return mean, 0.0
    sd = statistics.stdev(values)
    se = sd / math.sqrt(n)
    # z = 1.96 for 95% CI
    z = 1.96 if confidence == 0.95 else 2.576
    return mean, z * se


def _paired_ttest(a: List[float], b: List[float]) -> float:
    """Return p-value for paired t-test (a - b).  a and b must be same length.
    Uses normal approximation (which is fine for N >= 30)."""
    if len(a) != len(b) or len(a) < 2:
        return 1.0
    diffs = [a[i] - b[i] for i in range(len(a))]
    mean_diff = statistics.fmean(diffs)
    sd_diff = statistics.stdev(diffs)
    if sd_diff == 0:
        return 0.0 if mean_diff != 0 else 1.0
    t = mean_diff / (sd_diff / math.sqrt(len(diffs)))
    # Normal approximation (two-tailed): p ≈ 2 × (1 − Φ(|t|))
    # Abramowitz & Stegun 26.2.17 — accurate enough for our purposes
    x = abs(t)
    # erf approximation
    t_arg = 1.0 / (1.0 + 0.2316419 * x)
    d = 0.3989422804014327 * math.exp(-x * x / 2.0)
    prob = d * t_arg * (
        0.31938153 + t_arg * (
            -0.356563782 + t_arg * (
                1.781477937 + t_arg * (
                    -1.821255978 + t_arg * 1.330274429))))
    p_one_tail = prob
    return 2.0 * p_one_tail


def main():
    parser = argparse.ArgumentParser(
        description="Run many simulations to statistically compare "
                    "Nearest-Entrance vs Floor-Directed routing.")
    parser.add_argument("--runs", type=int, default=1000,
                        help="number of simulations per policy (default 1000)")
    parser.add_argument("--peak", type=int, default=60,
                        help="synthetic peak arrivals per hour (default 60)")
    parser.add_argument("--day-type", type=str, default="Saturday",
                        help="demand day type (default Saturday)")
    parser.add_argument("--out", type=str, default="baseline_study.csv",
                        help="output CSV path (default baseline_study.csv)")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="starting seed (default 0)")
    args = parser.parse_args()

    N = args.runs
    peak = args.peak
    day_type = args.day_type

    print(f"\n=== WPark Baseline Study ===")
    print(f"  Runs per policy:  {N}")
    print(f"  Peak arrivals/hr: {peak}")
    print(f"  Day type:         {day_type}")
    print(f"  Policies:         {', '.join(POLICIES)}")
    print()

    # Collect per-run metrics
    results: Dict[str, Dict[str, List[float]]] = {
        p: {k: [] for k in METRICS_TO_TRACK} for p in POLICIES
    }

    t0 = time.time()
    progress_every = max(1, N // 50)
    for i in range(N):
        seed = args.seed_offset + i + 1
        for policy in POLICIES:
            vals = _one_run(policy, peak, day_type, seed)
            for k, v in vals.items():
                results[policy][k].append(v)
        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            pct = (i + 1) / N * 100
            eta = elapsed / (i + 1) * (N - i - 1)
            print(f"  Progress: {i+1:4d}/{N}  ({pct:3.0f}%)   "
                  f"elapsed={elapsed:5.1f}s  eta={eta:5.1f}s", flush=True)

    total_time = time.time() - t0
    print(f"\n  Completed in {total_time:.1f}s ({total_time/N:.2f}s per run)\n")

    # ── Report ──
    display_fmt = {
        "avg_cruise_time":         ("s",  1),
        "avg_walk_time":           ("s",  1),
        "avg_total_wasted":        ("s",  1),
        "avg_queue_wait_seconds":  ("s",  2),
        "correct_floor_pct":       ("%",  1),
        "vehicles_served":         ("",   0),
        "vehicles_rejected":       ("",   0),
        "total_vehicles":          ("",   0),
        "total_extra_spend_daily": ("£",  2),
        "avg_extra_spend_per_car": ("£",  4),
    }

    header_line = f"{'METRIC':<26s}  " + "  ".join(
        f"{p:>22s}" for p in POLICIES
    )
    print("=" * len(header_line))
    print(header_line)
    print("-" * len(header_line))

    for k in METRICS_TO_TRACK:
        unit, dp = display_fmt[k]
        row = f"{k:<26s}  "
        for p in POLICIES:
            mean, ci = _mean_ci(results[p][k])
            cell = f"{mean:.{dp}f} ± {ci:.{dp}f} {unit}".strip()
            row += f"{cell:>22s}  "
        print(row.rstrip())
    print("=" * len(header_line))

    # Headline comparison: each policy vs baseline
    print("\n── HEADLINE DELTAS vs Nearest Entrance (baseline) ──")
    for p in POLICIES[1:]:
        print(f"\n  {p}:")
        for metric, label in [
            ("avg_cruise_time",        "cruise saved"),
            ("avg_walk_time",          "walk saved"),
            ("avg_total_wasted",       "TOTAL saved"),
            ("total_extra_spend_daily", "extra daily £"),
            ("correct_floor_pct",      "correct floor %"),
        ]:
            baseline = results["nearest_entrance"][metric]
            smart = results[p][metric]
            diffs = [baseline[i] - smart[i] for i in range(len(baseline))]
            if metric in ("total_extra_spend_daily", "correct_floor_pct"):
                # Higher is better — flip sign
                diffs = [-d for d in diffs]
            d_mean, d_ci = _mean_ci(diffs)
            p_value = _paired_ttest(baseline, smart)
            sig = "***" if p_value < 0.001 else (
                "**"  if p_value < 0.01 else (
                "*"   if p_value < 0.05 else "n.s."))
            suffix = ("s" if metric != "total_extra_spend_daily"
                      and metric != "correct_floor_pct"
                      else ("£" if "spend" in metric else "%"))
            print(f"    {label:20s}  {d_mean:+6.2f} ± {d_ci:.2f} {suffix}   ({sig})")

    # Rejection comparison
    print("\n  Rejection rates:")
    for p in POLICIES:
        rates = [
            (results[p]["vehicles_rejected"][i]
             / max(1, results[p]["total_vehicles"][i])) * 100
            for i in range(len(results[p]["vehicles_rejected"]))
        ]
        mean, ci = _mean_ci(rates)
        print(f"    {p:20s}  {mean:.1f}% ± {ci:.1f}%")

    # Write CSV
    out_path = os.path.join(HERE, args.out)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["run", "policy"] + list(METRICS_TO_TRACK)
        w.writerow(header)
        for policy in POLICIES:
            n_rows = len(results[policy][METRICS_TO_TRACK[0]])
            for i in range(n_rows):
                row = [i + 1, policy] + [results[policy][k][i] for k in METRICS_TO_TRACK]
                w.writerow(row)
    print(f"\n  Per-run data written to: {out_path}")
    print()


if __name__ == "__main__":
    main()
