"""Behavioural tests for the simulation engine and routing policies."""
import pytest

from carpark import build_demo_carpark
from demand import build_synthetic_demand
from engine import (
    SimulationEngine, classify_stay, _estimate_reward_for_bay,
    REWARD_WORST_CASE_WASTED, REWARD_CONVERSION_RATE,
    total_walk_seconds, Vehicle, _load_ppo_model,
)


def run_policy(policy, seed=7, peak=40):
    cp = build_demo_carpark()
    demand = build_synthetic_demand("Demo", peak_arrivals_per_hour=peak)
    eng = SimulationEngine(cp, demand, policy, "Saturday", random_seed=seed)
    eng.generate_arrivals()
    return eng.run(), eng


# ── Conservation & bookkeeping ──────────────────────────────────────────

def test_car_conservation():
    m, eng = run_policy("nearest_entrance")
    assert m.vehicles_served + m.vehicles_rejected <= m.total_vehicles
    assert m.vehicles_served == len(m.vehicles_log)
    assert m.vehicles_served == len([v for v in eng.all_vehicles if v.assigned_bay])


def test_no_bay_double_booking():
    """No two vehicles may occupy the same bay at overlapping times."""
    m, _ = run_policy("nearest_entrance")
    by_bay = {}
    for v in m.vehicles_log:
        by_bay.setdefault(v["assigned_bay"], []).append(
            (v["parked_from"], v["departure_second"]))
    for bay_id, intervals in by_bay.items():
        intervals.sort()
        for (s1, e1), (s2, e2) in zip(intervals, intervals[1:]):
            assert e1 <= s2, f"bay {bay_id}: overlapping occupancy " \
                             f"({s1}-{e1}) vs ({s2}-{e2})"


def test_timeline_ordering():
    m, _ = run_policy("floor_directed")
    for v in m.vehicles_log:
        assert v["arrival_second"] <= v["parked_from"]
        assert v["parked_from"] < v["departure_second"]
        assert v["departure_second"] <= v["gone_second"]


def test_determinism_same_seed():
    m1, _ = run_policy("greedy_smart", seed=13)
    m2, _ = run_policy("greedy_smart", seed=13)
    assert m1.avg_total_wasted == m2.avg_total_wasted
    assert m1.vehicles_served == m2.vehicles_served
    assert [v["assigned_bay"] for v in m1.vehicles_log] == \
           [v["assigned_bay"] for v in m2.vehicles_log]


def test_different_seeds_differ():
    m1, _ = run_policy("greedy_smart", seed=13)
    m2, _ = run_policy("greedy_smart", seed=14)
    assert m1.vehicles_served != m2.vehicles_served or \
           m1.avg_total_wasted != m2.avg_total_wasted


# ── Policy sanity ───────────────────────────────────────────────────────

def test_floor_match_beats_baseline_on_walking():
    mb, _ = run_policy("nearest_entrance")
    mf, _ = run_policy("floor_directed")
    assert mf.avg_walk_time < mb.avg_walk_time
    assert mf.correct_floor_pct > mb.correct_floor_pct


def test_greedy_beats_baseline_on_revenue():
    mb, _ = run_policy("nearest_entrance")
    mg, _ = run_policy("greedy_smart")
    assert mg.total_extra_spend_daily > mb.total_extra_spend_daily


# ── Reward function (the webapp's worked example) ───────────────────────

def test_reward_worked_example():
    """Short-stay high-spend customers must out-score long-stay low-spend
    customers for the same bay, in the exact proportions of the formula."""
    cp = build_demo_carpark()
    bay = cp.floors[0].bays[14]     # a ground-floor BOT-row bay
    grocery = Vehicle(id="a", arrival_second=12 * 3600,
                      duration_seconds=30 * 60, destination_floor=0,
                      destination_shop="Demo Supermarket")
    cinema = Vehicle(id="b", arrival_second=12 * 3600,
                     duration_seconds=150 * 60, destination_floor=2,
                     destination_shop="Demo Cinema")

    r_grocery = _estimate_reward_for_bay(grocery, bay, cp)
    r_cinema = _estimate_reward_for_bay(cinema, bay, cp)
    assert r_grocery > r_cinema

    # Hand-computed expectation for the grocery customer
    wasted = (bay.cruise_in_seconds + bay.cruise_out_seconds
              + total_walk_seconds(bay, 0))
    expected = (max(0.0, REWARD_WORST_CASE_WASTED - wasted)
                * REWARD_CONVERSION_RATE / 3600.0 * 22.0) / 30.0
    assert r_grocery == pytest.approx(expected)


def test_stay_classification_boundaries():
    assert classify_stay(59 * 60) == "short"
    assert classify_stay(60 * 60) == "medium"
    assert classify_stay(119 * 60) == "medium"
    assert classify_stay(120 * 60) == "long"


# ── PPO integration (regression for the silent-fallback bug) ────────────

def test_ppo_model_loads():
    """The shipped model MUST load. If this fails, every 'PPO' result in
    the study is actually greedy - the exact bug this repo once had."""
    assert _load_ppo_model() is not None


def test_ppo_differs_from_greedy():
    mg, _ = run_policy("greedy_smart", seed=21)
    mn, _ = run_policy("neural_smart", seed=21)
    greedy_bays = [v["assigned_bay"] for v in mg.vehicles_log]
    neural_bays = [v["assigned_bay"] for v in mn.vehicles_log]
    assert greedy_bays != neural_bays, \
        "PPO produced identical assignments to greedy - model not loading?"
