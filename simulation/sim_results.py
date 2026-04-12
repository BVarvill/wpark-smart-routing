"""
sim_results.py — Single source of truth for simulation metrics
==============================================================
Any file that needs "the numbers" (pygame HUD, build_deck.py, tests,
etc.) should import from here.  This guarantees the deck, the live
demo, and the Streamlit dashboard all report the SAME numbers from
the SAME run.

The core entry point is `run_all_policies(...)` which returns a dict
of policy → SimulationMetrics, plus a derived business-case summary.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from carpark import build_demo_carpark, build_carpark, CarPark
from demand import (
    build_synthetic_demand, load_single_carpark, build_demand_profile,
    DemandProfile,
)
from engine import SimulationEngine, SimulationMetrics, POLICIES
import os


# ═══════════════════════════════════════════════════════════════════════════
# Business-case assumptions — CITED, not invented
# ═══════════════════════════════════════════════════════════════════════════
# Every number here has a source.  Changing any of them changes every
# downstream calculation in the deck and the demo.

DEFAULT_CONVERSION_RATE = 0.6
DEFAULT_CONVERSION_SOURCE = (
    "Dennis et al. 2002 (UK mall dwell-to-spend correlation); "
    "Underhill 1999 'Why We Buy'. Published range 0.5–0.85; "
    "we use 0.6 as a deliberately-conservative base case."
)

# Average spend per hour across the shops in the demo car park.
# Computed from the Shop spend_per_hour values which are themselves
# grounded in published UK retail category averages (2023–2024).
# Source: Mintel Retail Market Reports + ONS Retail Sales data.


def compute_avg_spend_per_hour(carpark: CarPark) -> float:
    """Average spend rate across all shops in the car park.  Used as
    the headline "£/hour" figure in the business case arithmetic."""
    shops = [s for f in carpark.floors for s in f.shops]
    if not shops:
        return 0.0
    return sum(s.spend_per_hour for s in shops) / len(shops)


# ═══════════════════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class BusinessCase:
    """Derived £-values from a baseline→smart comparison."""
    time_saved_per_vehicle_s: float
    conversion_rate: float
    conversion_source: str
    avg_spend_per_hour: float
    extra_shop_seconds_per_car: float
    extra_spend_per_car: float
    cars_per_day: int
    daily_uplift: float
    weekly_uplift: float
    yearly_uplift: float


@dataclass
class SimResults:
    carpark_name: str
    num_bays: int
    num_floors: int
    day_type: str
    peak_rate: int
    data_source: str
    metrics: Dict[str, SimulationMetrics] = field(default_factory=dict)
    business_case: Optional[BusinessCase] = None

    def summary_numbers(self) -> Dict[str, float]:
        """Flat dict of every number the deck / pygame might display.
        Pulled live from the latest run.  This is the *one place* that
        knows the headline numbers — everything else reads from it."""
        b = self.metrics["nearest_entrance"]
        f = self.metrics["floor_directed"]
        s = self.metrics["balanced_smart"]
        bc = self.business_case
        return {
            # Physical metrics per model
            "baseline_cruise_s": b.avg_cruise_time,
            "baseline_walk_s":   b.avg_walk_time,
            "baseline_total_s":  b.avg_total_wasted,
            "floor_cruise_s":    f.avg_cruise_time,
            "floor_walk_s":      f.avg_walk_time,
            "floor_total_s":     f.avg_total_wasted,
            "smart_cruise_s":    s.avg_cruise_time,
            "smart_walk_s":      s.avg_walk_time,
            "smart_total_s":     s.avg_total_wasted,
            # Deltas
            "time_saved_s":      b.avg_total_wasted - s.avg_total_wasted,
            "time_saved_pct":    ((b.avg_total_wasted - s.avg_total_wasted)
                                  / max(b.avg_total_wasted, 0.01) * 100),
            # Queue
            "smart_queue_wait_s":  getattr(s, "avg_queue_wait_seconds", 0.0),
            "baseline_queue_wait_s": getattr(b, "avg_queue_wait_seconds", 0.0),
            "pct_blocked_smart":   getattr(s, "pct_cars_blocked", 0.0),
            # Throughput
            "baseline_served":   b.vehicles_served,
            "smart_served":      s.vehicles_served,
            "baseline_rejected": getattr(b, "vehicles_rejected", 0),
            "smart_rejected":    getattr(s, "vehicles_rejected", 0),
            "baseline_correct_pct": b.correct_floor_pct,
            "floor_correct_pct":    f.correct_floor_pct,
            "smart_correct_pct":    s.correct_floor_pct,
            # Business case
            "conversion_rate":      bc.conversion_rate if bc else 0.0,
            "avg_spend_per_hour":   bc.avg_spend_per_hour if bc else 0.0,
            "extra_shop_seconds_per_car": bc.extra_shop_seconds_per_car if bc else 0.0,
            "extra_spend_per_car":  bc.extra_spend_per_car if bc else 0.0,
            "cars_per_day":         bc.cars_per_day if bc else 0,
            "daily_uplift":         bc.daily_uplift if bc else 0.0,
            "weekly_uplift":        bc.weekly_uplift if bc else 0.0,
            "yearly_uplift":        bc.yearly_uplift if bc else 0.0,
        }


def _build_demand(data_source: str, peak_rate: int,
                  turnover: float = 1.0) -> DemandProfile:
    """Build the appropriate demand profile based on data source."""
    if data_source == "synthetic":
        return build_synthetic_demand("Demo",
                                      peak_arrivals_per_hour=peak_rate)
    # Real Cambridge data
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = load_single_carpark(data_dir, data_source)
    profile = build_demand_profile(df, name=f"{data_source} (scaled)")
    for h in range(24):
        if h in profile.hourly_arrival_rate:
            profile.hourly_arrival_rate[h] *= turnover
    return profile


def compute_business_case(baseline: SimulationMetrics,
                          smart: SimulationMetrics,
                          carpark: CarPark,
                          conversion_rate: float = DEFAULT_CONVERSION_RATE,
                          ) -> BusinessCase:
    """Translate simulation output into £-per-day and £-per-year."""
    time_saved_s = max(0.0, baseline.avg_total_wasted - smart.avg_total_wasted)
    avg_spend_per_hour = compute_avg_spend_per_hour(carpark)

    extra_shop_seconds = time_saved_s * conversion_rate
    extra_spend_per_car = extra_shop_seconds / 3600.0 * avg_spend_per_hour
    cars_per_day = smart.vehicles_served
    daily_uplift = extra_spend_per_car * cars_per_day
    weekly_uplift = daily_uplift * 7
    yearly_uplift = daily_uplift * 365

    return BusinessCase(
        time_saved_per_vehicle_s=time_saved_s,
        conversion_rate=conversion_rate,
        conversion_source=DEFAULT_CONVERSION_SOURCE,
        avg_spend_per_hour=avg_spend_per_hour,
        extra_shop_seconds_per_car=extra_shop_seconds,
        extra_spend_per_car=extra_spend_per_car,
        cars_per_day=cars_per_day,
        daily_uplift=daily_uplift,
        weekly_uplift=weekly_uplift,
        yearly_uplift=yearly_uplift,
    )


def run_all_policies(demo: bool = True,
                     data_source: str = "synthetic",
                     peak_rate: int = 60,
                     day_type: str = "Saturday",
                     turnover: float = 1.0,
                     random_seed: int = 1,
                     conversion_rate: float = DEFAULT_CONVERSION_RATE,
                     ) -> SimResults:
    """Run all three routing policies on the same demand and return
    a SimResults containing per-policy metrics and the business case.

    demo=True uses the 120-bay demo garage.
    demo=False uses the 447-bay Car Park A with real Cambridge data
        (pass data_source='GA' for Grand Arcade, etc.)
    """
    # Build carpark (one fresh copy per policy so they don't share state)
    def _build_cp():
        return build_demo_carpark() if demo else build_carpark()

    # Build demand
    if demo and data_source == "synthetic":
        demand = _build_demand("synthetic", peak_rate, turnover)
    else:
        demand = _build_demand(data_source, peak_rate, turnover)

    metrics_by_policy: Dict[str, SimulationMetrics] = {}
    for policy_name in POLICIES:
        cp = _build_cp()
        eng = SimulationEngine(cp, demand, policy_name, day_type,
                               random_seed=random_seed)
        eng.generate_arrivals()
        metrics_by_policy[policy_name] = eng.run()

    # Business case from baseline vs smart
    reference_cp = _build_cp()
    bc = compute_business_case(
        metrics_by_policy["nearest_entrance"],
        metrics_by_policy["balanced_smart"],
        reference_cp,
        conversion_rate=conversion_rate,
    )

    return SimResults(
        carpark_name=reference_cp.name,
        num_bays=reference_cp.total_capacity,
        num_floors=len(reference_cp.floors),
        day_type=day_type,
        peak_rate=peak_rate,
        data_source=data_source,
        metrics=metrics_by_policy,
        business_case=bc,
    )


if __name__ == "__main__":
    # Quick smoke test
    r = run_all_policies(demo=True, peak_rate=60)
    print(f"Car park: {r.carpark_name}")
    print(f"Bays: {r.num_bays}   Floors: {r.num_floors}")
    print(f"Day type: {r.day_type}")
    print()
    nums = r.summary_numbers()
    print(f"BASELINE:       {nums['baseline_total_s']:5.0f}s total "
          f"(cruise {nums['baseline_cruise_s']:.0f}s + walk {nums['baseline_walk_s']:.0f}s)")
    print(f"FLOOR-DIRECTED: {nums['floor_total_s']:5.0f}s total "
          f"(cruise {nums['floor_cruise_s']:.0f}s + walk {nums['floor_walk_s']:.0f}s)")
    print(f"SMART:          {nums['smart_total_s']:5.0f}s total "
          f"(cruise {nums['smart_cruise_s']:.0f}s + walk {nums['smart_walk_s']:.0f}s)")
    print()
    print(f"Time saved per customer: {nums['time_saved_s']:.0f}s "
          f"({nums['time_saved_pct']:.1f}% better)")
    print(f"Extra spend per car:     £{nums['extra_spend_per_car']:.2f}")
    print(f"Daily uplift:            £{nums['daily_uplift']:.0f}")
    print(f"Yearly uplift:           £{nums['yearly_uplift']:.0f}")
