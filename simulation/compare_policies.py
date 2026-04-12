"""
compare_policies.py — Side-by-side analysis of all routing policies
====================================================================
Run this to get a clear, printable comparison of every metric across
all policies on the SAME demand data (same seed = same customers).

Usage:
    python compare_policies.py                # default seed=1, peak=60
    python compare_policies.py --seed 42      # different random day
    python compare_policies.py --peak 80      # busier day
    python compare_policies.py --verbose       # per-hour breakdown

This is the tool you use to critically analyse what the models produce.
"""
import argparse, os, sys, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from carpark import build_demo_carpark
from demand import build_synthetic_demand
from engine import SimulationEngine, POLICIES

POLICY_NAMES = ["nearest_entrance", "floor_directed", "greedy_smart", "neural_smart"]
SHORT = {"nearest_entrance": "BASELINE", "floor_directed": "FLOOR-DIR",
         "greedy_smart": "GREEDY", "neural_smart": "NEURAL"}


def run_one(policy, peak, seed):
    cp = build_demo_carpark()
    demand = build_synthetic_demand("Demo", peak_arrivals_per_hour=peak)
    eng = SimulationEngine(cp, demand, policy, "Saturday", random_seed=seed)
    eng.generate_arrivals()
    return eng.run(), eng


def fmt(val, unit="s", dp=1):
    if unit == "s":
        return f"{val:.{dp}f}s"
    elif unit == "%":
        return f"{val:.{dp}f}%"
    elif unit == "£":
        return f"£{val:.2f}"
    return f"{val:.{dp}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--peak", type=int, default=60)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    results = {}
    for pol in POLICY_NAMES:
        m, eng = run_one(pol, args.peak, args.seed)
        results[pol] = m

    W = 18  # column width

    # ══════════════════════════════════════════════════════════════
    # SECTION 1: HEADLINE COMPARISON
    # ══════════════════════════════════════════════════════════════
    print()
    print("=" * 90)
    print(f"  POLICY COMPARISON  (seed={args.seed}, peak={args.peak}/hr, 60-bay demo)")
    print("=" * 90)

    header = f"{'METRIC':<28s}" + "".join(f"{SHORT[p]:>{W}s}" for p in POLICY_NAMES)
    print(header)
    print("-" * len(header))

    rows = [
        ("Avg cruise time",        "avg_cruise_time",        "s"),
        ("Avg walk time",          "avg_walk_time",          "s"),
        ("Avg TOTAL wasted",       "avg_total_wasted",       "s"),
        ("Avg queue wait",         "avg_queue_wait_seconds",  "s"),
        ("Max queue wait",         "max_queue_wait_seconds",  "s"),
        ("% cars that waited",     "pct_cars_blocked",        "%"),
        ("Correct floor %",        "correct_floor_pct",       "%"),
        ("Served",                 "vehicles_served",         ""),
        ("Rejected",               "vehicles_rejected",       ""),
        ("Extra £/day (vs worst)", "total_extra_spend_daily",  "£"),
        ("Extra £/car",            "avg_extra_spend_per_car",  "£"),
    ]

    for label, attr, unit in rows:
        vals = []
        for pol in POLICY_NAMES:
            v = getattr(results[pol], attr, 0)
            vals.append(v)
        # Find best value
        if attr in ("avg_cruise_time", "avg_walk_time", "avg_total_wasted",
                     "avg_queue_wait_seconds", "max_queue_wait_seconds",
                     "vehicles_rejected"):
            best_idx = vals.index(min(vals))
        else:
            best_idx = vals.index(max(vals))

        row = f"{label:<28s}"
        for i, v in enumerate(vals):
            cell = fmt(v, unit)
            if i == best_idx:
                cell = f"*{cell}*"  # mark the winner
            row += f"{cell:>{W}s}"
        print(row)

    print("-" * len(header))
    print("  * = best on that metric")

    # ══════════════════════════════════════════════════════════════
    # SECTION 2: DELTA vs BASELINE
    # ══════════════════════════════════════════════════════════════
    print()
    print("── IMPROVEMENT vs BASELINE ──")
    bl = results["nearest_entrance"]
    for pol in POLICY_NAMES[1:]:
        sm = results[pol]
        dt = bl.avg_total_wasted - sm.avg_total_wasted
        dw = bl.avg_walk_time - sm.avg_walk_time
        dc = bl.avg_cruise_time - sm.avg_cruise_time
        dq = bl.avg_queue_wait_seconds - sm.avg_queue_wait_seconds
        d_spend = sm.total_extra_spend_daily - bl.total_extra_spend_daily
        d_correct = sm.correct_floor_pct - bl.correct_floor_pct
        print(f"  {SHORT[pol]}:")
        print(f"    Total wasted:   {dt:+.1f}s  ({'better' if dt > 0 else 'worse'})")
        print(f"    Cruise:         {dc:+.1f}s")
        print(f"    Walk:           {dw:+.1f}s")
        print(f"    Queue wait:     {dq:+.1f}s")
        print(f"    Extra £/day:    {d_spend:+.0f}")
        print(f"    Correct floor:  {d_correct:+.1f}%")

    # ══════════════════════════════════════════════════════════════
    # SECTION 3: CONGESTION DETAIL
    # ══════════════════════════════════════════════════════════════
    print()
    print("── CONGESTION DETAIL ──")
    for pol in POLICY_NAMES:
        m = results[pol]
        waits = [v.get("queue_wait_s", 0) for v in m.vehicles_log]
        nonzero = [w for w in waits if w > 0]
        pct = len(nonzero) / max(1, len(waits)) * 100
        avg_nz = statistics.mean(nonzero) if nonzero else 0
        max_w = max(nonzero) if nonzero else 0
        total_w = sum(waits)
        print(f"  {SHORT[pol]:10s}  waited={len(nonzero):3d}/{len(waits):3d} ({pct:.0f}%)  "
              f"avg(waiters)={avg_nz:.0f}s  max={max_w:.0f}s  total={total_w:.0f}s")

    # ══════════════════════════════════════════════════════════════
    # SECTION 4: PER-HOUR BREAKDOWN (if verbose)
    # ══════════════════════════════════════════════════════════════
    if args.verbose:
        print()
        print("── PER-HOUR OCCUPANCY & ARRIVALS ──")
        print(f"{'Hour':<6s}" + "".join(
            f"  {SHORT[p]+' occ':>{W}s}" for p in POLICY_NAMES))
        for h in range(6, 23):
            row = f"{h:02d}:00 "
            for pol in POLICY_NAMES:
                m = results[pol]
                occ = sum(1 for v in m.vehicles_log
                          if v["arrival_second"] <= h * 3600
                          and v.get("departure_second", 0) > h * 3600)
                row += f"{occ:>{W}d}"
            print(row)

        print()
        print("── PER-HOUR AVERAGE QUEUE WAIT ──")
        print(f"{'Hour':<6s}" + "".join(
            f"  {SHORT[p]:>{W}s}" for p in POLICY_NAMES))
        for h in range(6, 23):
            row = f"{h:02d}:00 "
            for pol in POLICY_NAMES:
                m = results[pol]
                hour_waits = [v.get("queue_wait_s", 0) for v in m.vehicles_log
                              if h * 3600 <= v["arrival_second"] < (h + 1) * 3600]
                avg = statistics.mean(hour_waits) if hour_waits else 0
                row += f"{avg:>{W}.1f}"
            print(row)

    print()
    print("To run with per-hour breakdown: python compare_policies.py --verbose")
    print("To try a different day:         python compare_policies.py --seed 42")
    print()


if __name__ == "__main__":
    main()
