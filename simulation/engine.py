"""
Simulation Engine — Time-accurate discrete-event car park simulation
====================================================================
Replays real demand patterns from the 5 Cambridge car parks through the
WPark Car Park A model.  Every vehicle gets a real arrival TIME (in
seconds, not minutes) so the UI can step the clock one second at a time.

Key outputs per vehicle:
    arrival_second           — the moment the car reaches the entrance
    entry_travel_seconds     — time to drive in + park (congestion-adjusted)
    parked_from_second       — moment the car is fully parked
    departure_second         — moment the driver comes back to the bay
    exit_travel_seconds      — time to unpark + drive out (congestion-adjusted)
    gone_second              — moment the car has left the car park

Everything on the KPI page is derived from those fields.  Nothing is
random at render time — pick any second and the state is deterministic.
"""

import logging
import math
import os

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

from carpark import (
    CarPark, ParkingBay, BayStatus, Shop,
    WALK_SPEED_MPS, PARK_MANEUVER_SECONDS, STAIR_SECONDS_PER_FLOOR,
)
from demand import DemandProfile


# ════════════════════════════════════════════════════════════════════════════
# Data classes
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Vehicle:
    id: str
    arrival_second: int                         # seconds since midnight
    duration_seconds: float                     # shop dwell time
    destination_floor: int
    destination_shop: str

    # Filled in after policy assignment
    assigned_bay: Optional[str] = None
    assigned_floor: Optional[int] = None

    # Time-line (all in seconds since midnight)
    entry_travel_seconds: float = 0.0           # drive-in + park maneuver (with congestion)
    parked_from_second: Optional[int] = None    # arrival + entry_travel
    departure_second: Optional[int] = None      # parked_from + duration
    exit_travel_seconds: float = 0.0            # unpark + drive-out (with congestion)
    gone_second: Optional[int] = None           # departure + exit_travel

    # Derived metrics
    cruise_time_seconds: float = 0.0            # entry + exit travel time total
    walk_time_seconds: float = 0.0              # bay → shops (plus stair penalty)
    congestion_factor_in: float = 1.0
    congestion_factor_out: float = 1.0

    # Lane-segment queuing (Option A) — actual seconds spent waiting for
    # other cars to clear lane segments the driver had to traverse.
    queue_wait_seconds: float = 0.0

    # Cellular model — pre-computed by the tick stepper.  These are
    # populated before _arrive() runs, so _arrive just copies them.
    pre_computed_entry_travel: Optional[float] = None
    pre_computed_exit_travel:  Optional[float] = None
    pre_computed_wait_ticks: int = 0

    # Backward-compat fields used by older tabs
    walking_distance: float = 0.0
    cruising_distance: float = 0.0
    queue_wait_minutes: float = 0.0

    @property
    def arrival_minute(self) -> int:           # back-compat
        return self.arrival_second // 60


# ════════════════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════════════════

# Normalisation ceilings — raw value that corresponds to a score of 0.
NORM_CRUISE_SECONDS = 120.0
NORM_WALK_SECONDS   = 180.0
NORM_TOTAL_WASTED   = 300.0   # 5 min of not-parking time = 0 score


# Stay-length thresholds (seconds)
SHORT_STAY_MAX  = 60 * 60      # < 1 hour
MEDIUM_STAY_MAX = 120 * 60     # 1-2 hours


def classify_stay(duration_seconds: float) -> str:
    if duration_seconds < SHORT_STAY_MAX:
        return "short"
    if duration_seconds < MEDIUM_STAY_MAX:
        return "medium"
    return "long"


@dataclass
class SimulationMetrics:
    policy_name: str = ""
    total_vehicles: int = 0
    vehicles_served: int = 0
    vehicles_rejected: int = 0

    # Averages (seconds)
    avg_cruise_time: float = 0.0
    avg_walk_time: float = 0.0
    avg_total_wasted: float = 0.0            # cruise + walk
    avg_entry_travel: float = 0.0
    avg_exit_travel: float = 0.0
    avg_congestion_factor: float = 1.0
    # Option A lane-segment queuing
    avg_queue_wait_seconds: float = 0.0       # avg seconds spent stuck behind other cars
    max_queue_wait_seconds: float = 0.0       # worst case
    pct_cars_blocked: float = 0.0             # % of cars that had any queue wait

    # Peak / throughput
    peak_occupancy_pct: float = 0.0
    throughput_per_hour: float = 0.0
    max_concurrent_in_transit: int = 0

    # Legacy kept for back-compat
    avg_walking_distance: float = 0.0
    avg_cruising_distance: float = 0.0
    avg_queue_wait: float = 0.0
    occupancy_balance_score: float = 0.0

    # Time-series (5 min bins)
    floor_occupancy_over_time: Dict[int, List[float]] = field(default_factory=dict)
    total_occupancy_over_time: List[float] = field(default_factory=list)
    avg_cruise_time_by_hour: Dict[int, float] = field(default_factory=dict)
    avg_walk_time_by_hour: Dict[int, float] = field(default_factory=dict)
    in_transit_over_time: List[int] = field(default_factory=list)
    arrivals_per_hour: Dict[int, int] = field(default_factory=dict)
    departures_per_hour: Dict[int, int] = field(default_factory=dict)

    # Per-vehicle log
    vehicles_log: List[Dict] = field(default_factory=list)

    # Pre-computed frame snapshots for fast scrubbing (key = minute of day)
    frames: Dict[int, Dict] = field(default_factory=dict)

    # Stay-length breakdown
    stay_short_count: int = 0
    stay_medium_count: int = 0
    stay_long_count: int = 0

    # Correct floor %
    correct_floor_pct: float = 0.0

    # Economic outputs (computed later by the app vs a baseline)
    total_shop_time_minutes: float = 0.0    # total real shop minutes across all vehicles

    # Option-B revenue metric: extra spend captured by each served car,
    # summed across all served cars for the day.  Populated by _finalise().
    total_extra_spend_daily: float = 0.0
    avg_extra_spend_per_car: float = 0.0

    # ── Scores ────────────────────────────────────────────────────────
    def _clamp(self, x: float) -> float:
        return max(0.0, min(1.0, x))

    @property
    def score_cruise(self) -> float:
        return self._clamp(1.0 - self.avg_cruise_time / NORM_CRUISE_SECONDS)

    @property
    def score_walk(self) -> float:
        return self._clamp(1.0 - self.avg_walk_time / NORM_WALK_SECONDS)

    @property
    def score_total_wasted(self) -> float:
        """Single unified score — 1 means zero wasted time, 0 means 5+ minutes wasted."""
        return self._clamp(1.0 - self.avg_total_wasted / NORM_TOTAL_WASTED)

    def fitness(self, w_cruise: float = 0.5, w_walk: float = 0.5) -> float:
        total_w = max(w_cruise + w_walk, 1e-9)
        return (w_cruise * self.score_cruise + w_walk * self.score_walk) / total_w

    @property
    def fitness_score(self) -> float:
        return self.fitness()

    def breakdown(self, w_cruise: float = 0.5, w_walk: float = 0.5) -> dict:
        total_w = max(w_cruise + w_walk, 1e-9)
        return {
            "Cruising Time": {
                "score": self.score_cruise,
                "weight": w_cruise / total_w,
                "weighted": (w_cruise / total_w) * self.score_cruise,
                "raw": f"{self.avg_cruise_time:.1f} s",
            },
            "Walking Time": {
                "score": self.score_walk,
                "weight": w_walk / total_w,
                "weighted": (w_walk / total_w) * self.score_walk,
                "raw": f"{self.avg_walk_time:.1f} s",
            },
        }


# ════════════════════════════════════════════════════════════════════════════
# Helper — total walk time for a given bay + destination floor
# ════════════════════════════════════════════════════════════════════════════

def total_walk_seconds(bay: ParkingBay, destination_floor: int) -> float:
    """Base walk time from the bay to the nearest shop access point,
    plus 35 s of stair walking per floor they need to climb or descend."""
    floor_diff = abs(bay.floor - destination_floor)
    return bay.walk_seconds_to_shops + floor_diff * STAIR_SECONDS_PER_FLOOR


def total_cruise_seconds(bay: ParkingBay) -> float:
    """Base cruise-in + cruise-out time (congestion-free)."""
    return bay.cruise_in_seconds + bay.cruise_out_seconds


# ════════════════════════════════════════════════════════════════════════════
# Assignment policies
# ════════════════════════════════════════════════════════════════════════════

def policy_nearest_entrance(vehicle: Vehicle, carpark: CarPark, **kw) -> Optional[str]:
    """BASELINE — strict floor-by-floor filling, with 10% random noise.

    Simulates naive driver behaviour: 90% of the time, park in the FIRST
    empty bay you see as you drive through the car park.  Drivers see
    every ground-floor bay before any upper-floor bay, so the ground
    floor fills completely before anyone is sent upstairs.  Within a
    floor, pick the bay with the shortest drive-in distance.

    10% of the time, the driver "picks somewhere different" — they take a
    random available bay on a random floor.  This represents real-world
    noise: drivers who don't follow the obvious strategy, who get lost,
    or who spot a space they weren't expecting.

    This is the intentionally-dumb baseline.  It ignores the customer's
    destination shop entirely, so customers heading for upper-floor shops
    get parked on the ground floor and have to climb stairs.
    """
    rng = kw.get("rng")
    if rng is None:
        import random as _random
        rng = _random
    all_avail = carpark.get_all_available_bays()
    if not all_avail:
        return None
    # 10% random pick
    if rng.random() < 0.10:
        return rng.choice(all_avail).id
    # 90% strict floor-by-floor fill
    for floor_level in range(len(carpark.floors)):
        floor = carpark.get_floor(floor_level)
        if floor is None:
            continue
        available = [b for b in floor.bays if b.is_available()]
        if available:
            return min(available, key=lambda b: b.distance_to_entrance).id
    return None


def policy_floor_directed(vehicle: Vehicle, carpark: CarPark, **kw) -> Optional[str]:
    """FLOOR-DIRECTED — behaves like the baseline (nearest to entrance)
    but constrained to the destination floor.  Only if that floor is
    completely full does it fall back to the next-nearest floor."""
    # Preferred floor first, then by floor distance
    for fl in sorted(range(len(carpark.floors)),
                     key=lambda f: abs(f - vehicle.destination_floor)):
        floor = carpark.get_floor(fl)
        if floor is None:
            continue
        avail = [b for b in floor.bays if b.is_available()]
        if avail:
            return min(avail, key=lambda b: b.distance_to_entrance).id
    return None


def policy_balanced_smart(vehicle: Vehicle, carpark: CarPark,
                          concurrent_in_transit: int = 0, **kw) -> Optional[str]:
    """BALANCED SMART — jointly minimises (cruise + walk) time, aggressively
    load-balances across floors, and avoids pushing cars onto already-busy
    floors (which would amplify in-aisle congestion).

    Cost function per bay:
        cost(B) = cruise(B) × congestion_mult
                + walk(B, dest_floor)
                + floor_penalty(B.floor)
                + stair_bonus_if_correct_floor

    The floor penalty is non-linear so that as a floor approaches full,
    the system strongly prefers alternate floors."""
    available = carpark.get_all_available_bays()
    if not available:
        return None

    congestion_mult = 1.0 + 0.04 * concurrent_in_transit

    def occupancy_penalty(fl: int) -> float:
        floor = carpark.get_floor(fl)
        if floor is None:
            return 0.0
        pct = floor.occupancy_pct
        # Smooth quadratic penalty above 60% occupancy — grows quickly
        if pct < 60:
            return 0.0
        excess = (pct - 60) / 40.0          # 0 at 60%, 1.0 at 100%
        return 150.0 * excess ** 2          # up to 150 s penalty

    def cost(b: ParkingBay) -> float:
        cruise = (b.cruise_in_seconds + b.cruise_out_seconds) * congestion_mult
        walk   = total_walk_seconds(b, vehicle.destination_floor)
        return cruise + walk + occupancy_penalty(b.floor)

    return min(available, key=cost).id


# ─── Constants for the reward function (shared with rl_env.py) ──
# The reward function evaluates each (car, bay) assignment in £-per-minute-
# of-visit terms.  It's designed to favour customers whose visit time would
# be meaningfully impacted by time savings: short-stay high-spend shoppers
# get first pick, long-stay customers can absorb a longer walk.
REWARD_WORST_CASE_WASTED  = 400.0    # seconds — calibrated: roughly the slowest path
REWARD_CONVERSION_RATE    = 0.6      # extra-shop-time-to-spend conversion (cited)


def _find_shop(carpark: CarPark, shop_name: str) -> Optional[Shop]:
    """Look up a shop by name across all floors."""
    for f in carpark.floors:
        for s in f.shops:
            if s.name == shop_name:
                return s
    return None


def _estimate_reward_for_bay(vehicle: Vehicle, bay: ParkingBay,
                             carpark: CarPark) -> float:
    """Option-B reward: extra £ the mall would make from this customer
    per minute of their visit, if they were assigned to this bay.

        extra_shop_sec = max(0, WORST_CASE - (cruise + walk)) × CONVERSION
        extra_spend_£  = extra_shop_sec / 3600 × shop.spend_per_hour
        reward         = extra_spend_£ / visit_minutes

    Returns 0 if the shop can't be found or the visit is malformed.
    """
    shop = _find_shop(carpark, vehicle.destination_shop)
    if shop is None:
        return 0.0
    visit_sec = vehicle.duration_seconds or 0.0
    visit_min = visit_sec / 60.0
    if visit_min < 1.0:
        visit_min = 1.0
    # Physical "wasted" time for this (customer, bay) pair
    cruise = bay.cruise_in_seconds + bay.cruise_out_seconds
    walk   = total_walk_seconds(bay, vehicle.destination_floor)
    total_wasted = cruise + walk
    # Extra shopping time (positive if this bay is better than worst case)
    extra_wait_saved = max(0.0, REWARD_WORST_CASE_WASTED - total_wasted)
    extra_shop_seconds = extra_wait_saved * REWARD_CONVERSION_RATE
    extra_shop_hours = extra_shop_seconds / 3600.0
    extra_spend = extra_shop_hours * shop.spend_per_hour
    return extra_spend / visit_min


def policy_greedy_smart(vehicle: Vehicle, carpark: CarPark, **kw) -> Optional[str]:
    """GREEDY SMART — deterministic rule-based implementation of the Option B
    reward function.

    For each available bay, compute how much extra spend per minute of visit
    this customer would generate if assigned there, and pick the bay with
    the highest spend rate.  This is the analytical maximum of our reward
    function at each step — a contextual-bandit optimum if we had no
    congestion or sequential effects.
    """
    available = carpark.get_all_available_bays()
    if not available:
        return None
    # Rank by reward, break ties with shortest cruise+walk
    def sort_key(b):
        r = _estimate_reward_for_bay(vehicle, b, carpark)
        cw = b.cruise_in_seconds + b.cruise_out_seconds + total_walk_seconds(b, vehicle.destination_floor)
        return (-r, cw)
    available.sort(key=sort_key)
    return available[0].id


PPO_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "models", "ppo_policy.zip")

_PPO_MODEL = None
_PPO_LOADED = False


def _load_ppo_model():
    """Load the shipped MaskablePPO model once.  Failure to load is loud:
    the RL policy falling back to greedy would silently invalidate every
    'PPO vs greedy' comparison downstream."""
    global _PPO_MODEL, _PPO_LOADED
    if _PPO_LOADED:
        return _PPO_MODEL
    _PPO_LOADED = True
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as e:
        logger.error("[PPO] sb3-contrib not installed (%s) — "
                     "neural_smart will fall back to greedy_smart", e)
        return None
    if not os.path.exists(PPO_MODEL_PATH):
        logger.error("[PPO] Model not found at %s — "
                     "neural_smart will fall back to greedy_smart",
                     PPO_MODEL_PATH)
        return None
    _PPO_MODEL = MaskablePPO.load(PPO_MODEL_PATH)
    logger.info("[PPO] Loaded model from %s", PPO_MODEL_PATH)
    return _PPO_MODEL


def _build_ppo_obs(vehicle: Vehicle, carpark: CarPark) -> np.ndarray:
    """Build the 25-dim observation for the PPO model.

    KNOWN LIMITATION (documented in DECISIONS.md): the training env in
    rl_env.py tracks real per-floor assignment history, arrival rates and
    rejection counts.  The engine does not, so dims 3-5 and 17-18 are
    occupancy-derived proxies and dims 23-24 are neutral constants.  This
    is train/serve skew and one reason PPO only matches (rather than
    beats) the analytic greedy rule in the full simulation.
    """
    occ = []
    for f in carpark.floors:
        n_occ = sum(1 for b in f.bays if b.status == BayStatus.OCCUPIED)
        occ.append(n_occ / max(1, f.capacity))

    pressure = [min(1.0, o * 1.5) for o in occ]        # proxy for recent assignments

    dest = [0.0, 0.0, 0.0]
    dest[min(vehicle.destination_floor, 2)] = 1.0

    dur_min = vehicle.duration_seconds / 60.0
    stay = [1.0, 0.0, 0.0] if dur_min < 60 else (
           [0.0, 1.0, 0.0] if dur_min < 120 else [0.0, 0.0, 1.0])

    hour = (vehicle.arrival_second / 3600.0) % 24.0
    hour_sin = math.sin(2.0 * math.pi * hour / 24.0)
    hour_cos = math.cos(2.0 * math.pi * hour / 24.0)

    shop = _find_shop(carpark, vehicle.destination_shop)
    spend_norm = min(1.0, (shop.spend_per_hour if shop else 0) / 50.0)
    visit_norm = min(1.0, vehicle.duration_seconds / (240.0 * 60.0))

    total_occ = sum(occ) / 3.0
    rate_5 = min(1.0, total_occ * 2.0)                 # proxy for 5-min arrival rate
    rate_15 = min(1.0, total_occ * 1.5)                # proxy for 15-min arrival rate

    avail_per_floor = [1.0 - o for o in occ]
    is_peak = 1.0 if 11 <= hour <= 15 else 0.0
    rejected_norm = 0.0                                # not tracked in engine context
    gap_norm = 0.5                                     # neutral default

    return np.array(
        occ + pressure + dest + stay
        + [hour_sin, hour_cos, spend_norm, visit_norm, total_occ]
        + [rate_5, rate_15] + avail_per_floor
        + [is_peak, rejected_norm, gap_norm],
        dtype=np.float32,
    )


def policy_neural_smart(vehicle: Vehicle, carpark: CarPark, **kw) -> Optional[str]:
    """RL POLICY — MaskablePPO (stable-baselines3) over a 25-dim state.

    Trained from scratch (no warm-start, no imitation) on the Option-B
    revenue reward.  Falls back to greedy_smart if the model can't be
    loaded — loudly, via logger.error, because a silent fallback would
    mislabel greedy results as PPO results.
    """
    available = carpark.get_all_available_bays()
    if not available:
        return None

    model = _load_ppo_model()
    if model is not None:
        obs = _build_ppo_obs(vehicle, carpark)
        all_bays = [b for f in carpark.floors for b in f.bays]
        mask = np.array([b.is_available() for b in all_bays], dtype=bool)
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        action = int(action)
        if 0 <= action < len(all_bays) and mask[action]:
            return all_bays[action].id
        logger.warning("[PPO] Model chose invalid/masked action %d — "
                       "falling back to greedy for vehicle %s",
                       action, vehicle.id)

    return policy_greedy_smart(vehicle, carpark, **kw)


POLICIES = {
    "nearest_entrance": ("Baseline (no routing)",          policy_nearest_entrance),
    "floor_directed":   ("Floor-Match",                    policy_floor_directed),
    "balanced_smart":   ("Balanced Smart Routing",          policy_balanced_smart),
    "greedy_smart":     ("Revenue-Optimised",              policy_greedy_smart),
    "neural_smart":     ("RL Policy (PPO)",                policy_neural_smart),
}


# ════════════════════════════════════════════════════════════════════════════
# Simulation engine
# ════════════════════════════════════════════════════════════════════════════

class SimulationEngine:
    def __init__(self, carpark, demand, policy_name="nearest_entrance",
                 day_type="Saturday", random_seed=42):
        self.carpark = carpark
        self.demand = demand
        self.policy_name = policy_name
        self.policy_label, self.policy_fn = POLICIES[policy_name]
        self.day_type = day_type
        self.rng = np.random.RandomState(random_seed)

        self.vehicles: Dict[str, Vehicle] = {}          # currently parked
        self.all_vehicles: List[Vehicle] = []
        self.metrics = SimulationMetrics(policy_name=self.policy_label)
        self.arrival_schedule: List[Vehicle] = []

        # Active congestion tracking (updated every minute)
        self._in_transit: List[Vehicle] = []
        self._max_concurrent: int = 0

        # ── Option A: Lane-segment queuing state ────────────────────────
        # For each lane segment, the earliest time (in seconds since
        # midnight) at which a new car can *start* traversing it.
        # Only populated if the car park has entry_segments on its bays.
        self._segment_free_at: Dict[str, float] = {}
        try:
            from carpark import DEMO_LANE_SEGMENTS
            self._lane_segments = DEMO_LANE_SEGMENTS
        except ImportError:
            self._lane_segments = {}

    # ── Arrival schedule generation ──────────────────────────────────────
    def generate_arrivals(self):
        self.arrival_schedule = []
        all_shops = [s for f in self.carpark.floors for s in f.shops]

        for hour in range(24):
            rate = self.demand.get_arrival_rate(hour, self.day_type)
            n = self.rng.poisson(rate)
            for _ in range(n):
                # Random second inside the hour
                sec = hour * 3600 + self.rng.randint(0, 3600)
                dur_min = self.demand.sample_duration(rng=self.rng)
                shop = self._pick_shop(all_shops, hour)
                vid = f"v{hour:02d}_{_:03d}_{self.rng.randint(0, 9999):04d}"
                self.arrival_schedule.append(Vehicle(
                    id=vid,
                    arrival_second=sec,
                    duration_seconds=dur_min * 60.0,
                    destination_floor=shop.floor,
                    destination_shop=shop.name,
                ))
        self.arrival_schedule.sort(key=lambda v: v.arrival_second)

    def _pick_shop(self, shops, hour):
        w = np.array([3.0 if hour in s.peak_hours else 1.0 for s in shops])
        w /= w.sum()
        return shops[self.rng.choice(len(shops), p=w)]

    # ── Cellular pre-pass — exact tick-based physical simulation ────────
    def _run_cellular_prepass(self):
        """Discrete-time cellular-automaton pre-pass for demo car parks.

        Runs the entire day one second at a time.  At each tick:
          1. Spawn any cars whose arrival_second == tick (place on entry cell
             if free; else buffer them to retry next tick).
          2. For each active car in arrival order:
             - If parking: decrement parking_timer; at 0 release the cell.
             - If ready-to-depart (parked and departure time reached):
               try to move into approach cell for exit (if empty).
             - If moving (entering or exiting): try to advance one cell
               along the pre-computed path.  If the next cell is blocked,
               increment wait_ticks.
          3. Log each car's entry_travel_seconds, exit_travel_seconds,
             and wait_ticks.

        Result: every vehicle gets populated fields consistent with the
        cellular model.  _arrive() will just copy these pre-computed values.
        """
        # Import here to avoid a circular dependency
        from carpark import LaneCell, BayStatus as _BS  # noqa: F401

        if not self.carpark.lanes:
            return  # Not a cellular car park (full-scale Car Park A)

        # Reset all cells' occupancy
        for lane in self.carpark.lanes.values():
            for cell in lane.cells:
                cell.car_id = None

        # Reset all bays — we'll assign them dynamically during the
        # tick loop so that bays freed by departing cars become
        # available to later arrivals.
        for f in self.carpark.floors:
            for b in f.bays:
                b.status = BayStatus.AVAILABLE
                b.occupied_by = None

        # Car state tracking
        class CarState:
            __slots__ = ("v", "bay", "state", "path", "path_idx",
                         "parking_timer", "wait_ticks", "move_cooldown",
                         "enter_tick", "parked_tick", "depart_tick", "gone_tick")
            def __init__(self, v, bay):
                self.v = v
                self.bay = bay
                self.state = "waiting"   # waiting | entering | parking | parked | exiting | gone
                self.path = bay.entry_path_cells
                self.path_idx = 0
                self.parking_timer = 0
                self.wait_ticks = 0
                self.move_cooldown = 0   # ticks remaining before car can advance to next cell
                self.enter_tick = None
                self.parked_tick = None
                self.depart_tick = None
                self.gone_tick = None

        all_cars: List[CarState] = []            # full log (including skipped)
        waiting_to_enter: List[CarState] = []    # cars with assigned bays, waiting for entry cell
        active: List[CarState] = []               # cars currently inside the car park
        parked: List[CarState] = []               # cars fully parked

        END_TICK = 24 * 3600     # simulate full day
        arrival_idx = 0
        arrivals_sorted = sorted(self.arrival_schedule, key=lambda x: x.arrival_second)

        for tick in range(0, END_TICK):
            # ── 1. Dynamically assign bays to newly-arriving cars ──
            # Call the policy at the car's arrival tick, so it sees the
            # CURRENT state of the car park (bays freed by departed cars
            # are re-available).
            while (arrival_idx < len(arrivals_sorted) and
                   arrivals_sorted[arrival_idx].arrival_second <= tick):
                v = arrivals_sorted[arrival_idx]
                arrival_idx += 1
                bay_id = self.policy_fn(v, self.carpark,
                                        concurrent_in_transit=len(active),
                                        rng=self.rng)
                if bay_id is None:
                    # Park is full — car drives away.  Tracked as a
                    # rejection.  Rejections do NOT contribute to the
                    # time-based averages (those only iterate served
                    # cars) but they ARE counted for throughput analysis.
                    v.assigned_bay = None
                    self.metrics.vehicles_rejected += 1
                    continue
                bay = self.carpark.get_bay(bay_id)
                bay.status = BayStatus.OCCUPIED
                bay.occupied_by = v.id
                v.assigned_bay = bay_id
                v.assigned_floor = bay.floor
                car = CarState(v, bay)
                all_cars.append(car)
                waiting_to_enter.append(car)

            # Try to move waiting cars onto the entry cell
            still_waiting = []
            for car in waiting_to_enter:
                entry_cell = car.path[0] if car.path else None
                if entry_cell is not None and entry_cell.car_id is None:
                    entry_cell.car_id = car.v.id
                    car.state = "entering"
                    car.enter_tick = tick
                    active.append(car)
                else:
                    car.wait_ticks += 1
                    still_waiting.append(car)
            waiting_to_enter = still_waiting

            # ── 2. Check for departures from parked cars ──
            still_parked = []
            for car in parked:
                if car.state == "parked" and tick >= car.depart_tick:
                    car.state = "ready_to_exit"
                    car.path = car.bay.exit_path_cells
                    car.path_idx = 0
                    active.append(car)      # move back into active rotation
                else:
                    still_parked.append(car)
            parked = still_parked

            # ── 3. Advance active cars one cell ──
            # Process in arrival order (earlier arrivals get priority)
            active.sort(key=lambda c: c.v.arrival_second)
            new_active = []
            for car in active:
                if car.state == "parking":
                    car.parking_timer -= 1
                    if car.parking_timer <= 0:
                        # Release the approach cell — car is now "in the bay"
                        car.bay.approach_cell.car_id = None
                        car.state = "parked"
                        car.parked_tick = tick
                        car.depart_tick = tick + int(car.v.duration_seconds)
                        parked.append(car)
                        continue  # remove from active
                    new_active.append(car)
                    continue

                if car.state == "ready_to_exit":
                    # Car is in the bay (not in any cell).  Try to move
                    # into the approach cell (exit_path[0]).
                    approach = car.bay.approach_cell
                    if approach.car_id is None:
                        approach.car_id = car.v.id
                        car.state = "exiting"
                        car.path_idx = 0    # at exit_path[0] = approach cell
                        car.bay.status = BayStatus.AVAILABLE
                        car.bay.occupied_by = None
                    else:
                        car.wait_ticks += 1
                    new_active.append(car)
                    continue

                # entering or exiting — try to advance to next cell
                if car.path_idx >= len(car.path) - 1:
                    if car.state == "entering":
                        # Reached destination approach cell — start parking
                        car.state = "parking"
                        car.parking_timer = int(PARK_MANEUVER_SECONDS)
                        new_active.append(car)
                    else:  # exiting
                        # Last cell of exit path — release and gone
                        if car.path[car.path_idx].car_id == car.v.id:
                            car.path[car.path_idx].car_id = None
                        car.state = "gone"
                        car.gone_tick = tick
                    continue

                # TICKS_PER_CELL: a car takes 2 ticks to cross one cell
                # (5.85m per cell ÷ 3.33 m/s ≈ 1.76s → rounded to 2).
                # This makes the cellular model's time scale match
                # physical reality so congestion times are accurate.
                TICKS_PER_CELL = 2
                if car.move_cooldown > 0:
                    car.move_cooldown -= 1
                    new_active.append(car)
                    continue

                next_cell = car.path[car.path_idx + 1]
                if next_cell.car_id is None:
                    # Advance to next cell and start cooldown
                    cur = car.path[car.path_idx]
                    if cur.car_id == car.v.id:
                        cur.car_id = None
                    next_cell.car_id = car.v.id
                    car.path_idx += 1
                    car.move_cooldown = TICKS_PER_CELL - 1  # wait 1 more tick before next advance
                    new_active.append(car)
                else:
                    car.wait_ticks += 1
                    new_active.append(car)

            active = new_active

            # Stop early if nothing left to simulate after 24:00-ish
            if (arrival_idx >= len(arrivals_sorted)
                    and not active and not waiting_to_enter
                    and not any(c.state == "parked" for c in parked)
                    and tick > END_TICK - 3600):
                break

        # ── Finalise: stamp pre-computed fields on each vehicle ──
        for cs in all_cars:
            if cs.enter_tick is None:
                # Never got to enter — treat as rejected
                cs.v.assigned_bay = None
                continue
            v = cs.v
            v.pre_computed_wait_ticks = cs.wait_ticks

            # PROPER congestion calculation:
            # congestion = (actual entry time) − (free-flow entry time)
            # Free-flow = (n_cells − 1) advances × TICKS_PER_CELL + parking
            # The −1 is because the car starts AT cell 0 (no advance needed
            # for the first cell).
            TICKS_PER_CELL = 2
            n_entry_cells = len(cs.bay.entry_path_cells) if cs.bay.entry_path_cells else 0
            free_flow_entry = max(0, n_entry_cells - 1) * TICKS_PER_CELL + PARK_MANEUVER_SECONDS
            actual_entry = float((cs.parked_tick or cs.enter_tick) - v.arrival_second)
            v.queue_wait_seconds = max(0.0, actual_entry - free_flow_entry)
            v.pre_computed_entry_travel = float(
                (cs.parked_tick or cs.enter_tick) - v.arrival_second
            )
            if cs.gone_tick is not None and cs.depart_tick is not None:
                v.pre_computed_exit_travel = float(cs.gone_tick - cs.depart_tick)
            else:
                v.pre_computed_exit_travel = cs.bay.cruise_out_seconds
            v.parked_from_second = (cs.parked_tick
                                    if cs.parked_tick is not None
                                    else v.arrival_second)
            v.departure_second   = v.parked_from_second + int(v.duration_seconds)
            v.gone_second = (cs.gone_tick
                             if cs.gone_tick is not None
                             else v.departure_second + int(cs.bay.cruise_out_seconds))

        # Reset bays and cells — the main run() metric loop will re-occupy
        # bays as it iterates through arrivals via _arrive().
        for f in self.carpark.floors:
            for b in f.bays:
                b.status = BayStatus.AVAILABLE
                b.occupied_by = None
        for lane in self.carpark.lanes.values():
            for cell in lane.cells:
                cell.car_id = None

    # ── Main run loop — minute-level but arrivals keep second-precision ──
    def run(self) -> SimulationMetrics:
        self.carpark.reset()
        self.vehicles.clear()
        self.all_vehicles.clear()
        self._in_transit = []
        self._max_concurrent = 0
        self._segment_free_at = {}   # reset lane-segment queuing state
        self.metrics = SimulationMetrics(policy_name=self.policy_label)
        for f in self.carpark.floors:
            self.metrics.floor_occupancy_over_time[f.level] = []
        if not self.arrival_schedule:
            self.generate_arrivals()
        self.metrics.total_vehicles = len(self.arrival_schedule)

        # ── Cellular pre-pass for demo car parks ──
        # If this is a cellular car park (has .lanes), run the tick-based
        # physical simulation FIRST to pre-compute every vehicle's actual
        # cell-by-cell timeline.  Then the main metric loop below reads
        # those pre-computed values via _arrive().
        if self.carpark.lanes:
            self._run_cellular_prepass()

        # We still tick minute-by-minute (fast) but each arrival has its
        # own exact arrival_second.
        arrival_idx = 0
        hour_cruise_accum: Dict[int, List[float]] = defaultdict(list)
        hour_walk_accum:   Dict[int, List[float]] = defaultdict(list)

        for minute in range(360, 1440):          # simulate 06:00 – 24:00
            sec_now = minute * 60

            # ── Departures ───────────────────────────────────────────
            departing_ids = [vid for vid, v in self.vehicles.items()
                             if v.departure_second is not None
                             and v.departure_second <= sec_now]
            for vid in departing_ids:
                self._depart(vid)
                h = minute // 60
                self.metrics.departures_per_hour[h] = \
                    self.metrics.departures_per_hour.get(h, 0) + 1

            # ── Arrivals in this minute ──────────────────────────────
            while (arrival_idx < len(self.arrival_schedule) and
                   self.arrival_schedule[arrival_idx].arrival_second < sec_now + 60):
                v = self.arrival_schedule[arrival_idx]
                self._arrive(v)
                h = minute // 60
                self.metrics.arrivals_per_hour[h] = \
                    self.metrics.arrivals_per_hour.get(h, 0) + 1
                if v.assigned_bay:
                    hour_cruise_accum[h].append(v.cruise_time_seconds)
                    hour_walk_accum[h].append(v.walk_time_seconds)
                arrival_idx += 1

            # ── Concurrent-in-transit tracking (congestion) ─────────
            self._in_transit = [
                v for v in self._in_transit
                if v.gone_second is not None and v.gone_second > sec_now
                and (v.parked_from_second is None or v.parked_from_second > sec_now
                     or (v.departure_second is not None and sec_now >= v.departure_second))
            ]
            n_transit = len(self._in_transit)
            if n_transit > self._max_concurrent:
                self._max_concurrent = n_transit

            # ── 5-min snapshot ───────────────────────────────────────
            if minute % 5 == 0:
                for f in self.carpark.floors:
                    self.metrics.floor_occupancy_over_time[f.level].append(f.occupancy_pct)
                self.metrics.total_occupancy_over_time.append(self.carpark.overall_occupancy_pct)
                self.metrics.in_transit_over_time.append(n_transit)

                # Frame snapshot — key = minute of day.  Used by the UI
                # to scrub/play through the day in O(1) time.
                occupied_bays = {}
                for f in self.carpark.floors:
                    for b in f.bays:
                        if b.status == BayStatus.OCCUPIED:
                            occupied_bays[b.id] = b.occupied_by
                # Parked vehicles snapshot (light — just id + floor + dest)
                parked_snap = [
                    {
                        "id": v.id,
                        "bay": v.assigned_bay,
                        "assigned_floor": v.assigned_floor,
                        "dest_floor": v.destination_floor,
                        "dest_shop": v.destination_shop,
                        "walk_time": v.walk_time_seconds,
                        "cruise_time": v.cruise_time_seconds,
                    }
                    for v in self.vehicles.values()
                ]
                self.metrics.frames[minute] = {
                    "occupied_bays": occupied_bays,
                    "parked": parked_snap,
                    "in_transit_count": n_transit,
                }

        # Finalise hour-bucketed averages
        for h in range(24):
            if hour_cruise_accum[h]:
                self.metrics.avg_cruise_time_by_hour[h] = float(np.mean(hour_cruise_accum[h]))
            if hour_walk_accum[h]:
                self.metrics.avg_walk_time_by_hour[h] = float(np.mean(hour_walk_accum[h]))

        self._finalise()
        return self.metrics

    # ── Arrival handling ─────────────────────────────────────────────────
    def _compute_queue_wait_entry(self, bay, arrival_second: int) -> float:
        """Walk the car through its entry segments, accumulating wait time
        when a segment is blocked by another car ahead of it.  Updates
        self._segment_free_at as the car passes through.  Returns the
        total wait in seconds (0 if the path was clear)."""
        if not bay.entry_segments or not self._lane_segments:
            return 0.0

        total_wait = 0.0
        t = float(arrival_second)
        for seg_id in bay.entry_segments:
            seg = self._lane_segments.get(seg_id)
            if seg is None:
                continue
            free_at = self._segment_free_at.get(seg_id, 0.0)
            if free_at > t:
                wait = free_at - t
                total_wait += wait
                t = free_at
            # Car now enters this segment and occupies it for passage_seconds
            t += seg.passage_seconds
            self._segment_free_at[seg_id] = t

        # The LAST segment in entry_segments is the "approach lane" right
        # next to the bay.  While the car is backing in to park, it is
        # physically sticking out of this lane — block the lane for
        # PARK_MANEUVER_SECONDS so the car behind has to wait.
        last_seg_id = bay.entry_segments[-1]
        if last_seg_id in self._lane_segments:
            self._segment_free_at[last_seg_id] = t + PARK_MANEUVER_SECONDS

        return total_wait

    def _compute_queue_wait_exit(self, bay, departure_second: int) -> float:
        """Exit queuing is NOT tracked with the global segment_free_at
        dictionary because we're still processing arrivals at this point;
        projecting exit reservations into the future would make arriving
        cars wait incorrectly.  Instead return 0 — entry congestion is
        the bottleneck anyway (the "stuck behind a parker" effect)."""
        return 0.0

    def _arrive(self, v: Vehicle):
        # CELLULAR MODE: if the carpark has lanes, the pre-pass already
        # ran the physical simulation.  Either the car got a pre-computed
        # timeline (fast path) or it was rejected during the pre-pass
        # (already counted — skip silently, don't fall through to legacy).
        if self.carpark.lanes:
            if v.pre_computed_entry_travel is None:
                return    # pre-pass couldn't assign a bay; already counted
        if v.pre_computed_entry_travel is not None:
            bay_id = v.assigned_bay
            if bay_id is None:
                # Could not find a bay during pre-pass — skip silently,
                # don't count as "rejected" (rejected arrivals confuse
                # baseline vs smart comparison).
                return
            bay = self.carpark.get_bay(bay_id)
            bay.status = BayStatus.OCCUPIED
            bay.occupied_by = v.id
            v.assigned_floor = bay.floor
            v.congestion_factor_in = 1.0
            v.entry_travel_seconds = v.pre_computed_entry_travel
            v.exit_travel_seconds  = v.pre_computed_exit_travel or bay.cruise_out_seconds
            # parked_from_second / departure_second / gone_second are already
            # set by the pre-pass
            v.cruise_time_seconds = v.entry_travel_seconds + v.exit_travel_seconds
            v.walk_time_seconds   = total_walk_seconds(bay, v.destination_floor)
            v.cruising_distance   = bay.distance_to_entrance + bay.distance_to_exit
            floor_diff            = abs(bay.floor - v.destination_floor)
            v.walking_distance    = bay.distance_to_shops + floor_diff * WALK_SPEED_MPS * STAIR_SECONDS_PER_FLOOR
            v.queue_wait_minutes  = v.queue_wait_seconds / 60.0
            self.vehicles[v.id] = v
            self.all_vehicles.append(v)
            self._in_transit.append(v)
            self.metrics.vehicles_served += 1
            stay_class = classify_stay(v.duration_seconds)
            if stay_class == "short":
                self.metrics.stay_short_count += 1
            elif stay_class == "medium":
                self.metrics.stay_medium_count += 1
            else:
                self.metrics.stay_long_count += 1
            # Append to the vehicles_log — used by KPIs, the renderer,
            # and the business-case calculation.
            self.metrics.vehicles_log.append({
                "id": v.id,
                "arrival_second":     v.arrival_second,
                "duration_seconds":   v.duration_seconds,
                "dest_floor":         v.destination_floor,
                "dest_shop":          v.destination_shop,
                "assigned_bay":       bay_id,
                "assigned_floor":     bay.floor,
                "entry_travel":       v.entry_travel_seconds,
                "exit_travel":        v.exit_travel_seconds,
                "parked_from":        v.parked_from_second,
                "departure_second":   v.departure_second,
                "gone_second":        v.gone_second,
                "cruise_time":        v.cruise_time_seconds,
                "walk_time":          v.walk_time_seconds,
                "total_wasted":       v.cruise_time_seconds + v.walk_time_seconds,
                "queue_wait_s":       v.queue_wait_seconds,
                "stay_class":         stay_class,
                "walking_dist":       v.walking_distance,
                "cruising_dist":      v.cruising_distance,
                "congestion":         v.congestion_factor_in,
                "on_correct_floor":   bay.floor == v.destination_floor,
                "arrival":            v.arrival_second // 60,
                "duration":           v.duration_seconds / 60,
                "queue_wait":         v.queue_wait_minutes,
            })
            return

        # LEGACY MODE (full-scale Car Park A): old segment-FIFO path
        bay_id = self.policy_fn(v, self.carpark,
                                concurrent_in_transit=len(self._in_transit),
                                rng=self.rng)
        if bay_id is None:
            # Silently skip — we do not count "rejected" arrivals.
            return

        bay = self.carpark.get_bay(bay_id)
        bay.status = BayStatus.OCCUPIED
        bay.occupied_by = v.id
        v.assigned_bay = bay_id
        v.assigned_floor = bay.floor

        # Congestion is now modelled ONLY via the lane-segment FIFO queue
        # (no separate global multiplier — it was double-counting).  The
        # strict FIFO enforces "no overtaking" — a car arriving behind
        # another on the same lane must wait for that lane to clear.
        v.congestion_factor_in = 1.0

        queue_wait = self._compute_queue_wait_entry(bay, v.arrival_second)
        v.queue_wait_seconds = queue_wait

        v.entry_travel_seconds = bay.cruise_in_seconds + queue_wait
        v.parked_from_second = v.arrival_second + int(v.entry_travel_seconds)
        v.departure_second   = v.parked_from_second + int(v.duration_seconds)

        # Exit travel — same FIFO model, no multiplier
        exit_wait = self._compute_queue_wait_exit(bay, v.departure_second)
        v.exit_travel_seconds = bay.cruise_out_seconds + exit_wait
        v.queue_wait_seconds += exit_wait
        v.gone_second = v.departure_second + int(v.exit_travel_seconds)

        # Derived metric fields
        v.cruise_time_seconds = v.entry_travel_seconds + v.exit_travel_seconds
        v.walk_time_seconds   = total_walk_seconds(bay, v.destination_floor)

        # Legacy fields
        v.cruising_distance   = bay.distance_to_entrance + bay.distance_to_exit
        floor_diff            = abs(bay.floor - v.destination_floor)
        v.walking_distance    = bay.distance_to_shops + floor_diff * WALK_SPEED_MPS * STAIR_SECONDS_PER_FLOOR
        v.queue_wait_minutes  = v.queue_wait_seconds / 60.0

        self.vehicles[v.id] = v
        self.all_vehicles.append(v)
        self._in_transit.append(v)
        self.metrics.vehicles_served += 1

        stay_class = classify_stay(v.duration_seconds)
        if stay_class == "short":
            self.metrics.stay_short_count += 1
        elif stay_class == "medium":
            self.metrics.stay_medium_count += 1
        else:
            self.metrics.stay_long_count += 1

        self.metrics.vehicles_log.append({
            "id": v.id,
            "arrival_second":     v.arrival_second,
            "duration_seconds":   v.duration_seconds,
            "dest_floor":         v.destination_floor,
            "dest_shop":          v.destination_shop,
            "assigned_bay":       bay_id,
            "assigned_floor":     bay.floor,
            "entry_travel":       v.entry_travel_seconds,
            "exit_travel":        v.exit_travel_seconds,
            "parked_from":        v.parked_from_second,
            "departure_second":   v.departure_second,
            "gone_second":        v.gone_second,
            "cruise_time":        v.cruise_time_seconds,
            "walk_time":          v.walk_time_seconds,
            "total_wasted":       v.cruise_time_seconds + v.walk_time_seconds,
            "queue_wait_s":       v.queue_wait_seconds,    # Option A: actual queue wait
            "stay_class":         stay_class,
            "walking_dist":       v.walking_distance,
            "cruising_dist":      v.cruising_distance,
            "congestion":         v.congestion_factor_in,
            "on_correct_floor":   bay.floor == v.destination_floor,
            # Back-compat
            "arrival":            v.arrival_second // 60,
            "duration":           v.duration_seconds / 60,
            "queue_wait":         v.queue_wait_minutes,
        })

    def _depart(self, vid):
        v = self.vehicles.pop(vid, None)
        if v and v.assigned_bay:
            bay = self.carpark.get_bay(v.assigned_bay)
            if bay:
                bay.status = BayStatus.AVAILABLE
                bay.occupied_by = None

    # ── Finalisation ─────────────────────────────────────────────────────
    def _finalise(self):
        served = [v for v in self.all_vehicles if v.assigned_bay]
        if served:
            self.metrics.avg_cruise_time    = float(np.mean([v.cruise_time_seconds for v in served]))
            self.metrics.avg_walk_time      = float(np.mean([v.walk_time_seconds for v in served]))
            self.metrics.avg_total_wasted   = self.metrics.avg_cruise_time + self.metrics.avg_walk_time
            self.metrics.avg_entry_travel   = float(np.mean([v.entry_travel_seconds for v in served]))
            self.metrics.avg_exit_travel    = float(np.mean([v.exit_travel_seconds for v in served]))
            self.metrics.avg_congestion_factor = float(np.mean([v.congestion_factor_in for v in served]))

            # Lane-segment queue-wait metrics (Option A)
            queue_waits = [v.queue_wait_seconds for v in served]
            self.metrics.avg_queue_wait_seconds = float(np.mean(queue_waits)) if queue_waits else 0.0
            self.metrics.max_queue_wait_seconds = float(np.max(queue_waits)) if queue_waits else 0.0
            blocked = sum(1 for w in queue_waits if w > 0.5)
            self.metrics.pct_cars_blocked = (blocked / len(queue_waits) * 100.0) if queue_waits else 0.0

            # Total real shop dwell time across the day (minutes)
            self.metrics.total_shop_time_minutes = float(np.sum([v.duration_seconds for v in served])) / 60.0

            # Legacy averages
            self.metrics.avg_walking_distance  = float(np.mean([v.walking_distance for v in served]))
            self.metrics.avg_cruising_distance = float(np.mean([v.cruising_distance for v in served]))
            self.metrics.avg_queue_wait        = float(np.mean([v.queue_wait_minutes for v in served]))

        if self.metrics.total_occupancy_over_time:
            self.metrics.peak_occupancy_pct = max(self.metrics.total_occupancy_over_time)

        # Occupancy balance (legacy)
        if self.metrics.floor_occupancy_over_time:
            scores = []
            n = min(len(v) for v in self.metrics.floor_occupancy_over_time.values())
            for i in range(n):
                occs = [self.metrics.floor_occupancy_over_time[fl][i]
                        for fl in self.metrics.floor_occupancy_over_time]
                if np.mean(occs) > 5:
                    scores.append(max(0, 1.0 - np.std(occs) / 50.0))
            if scores:
                self.metrics.occupancy_balance_score = float(np.mean(scores))

        if self.metrics.vehicles_log:
            correct = sum(1 for v in self.metrics.vehicles_log if v.get("on_correct_floor"))
            total = len(self.metrics.vehicles_log)
            self.metrics.correct_floor_pct = correct / total * 100

        # Option-B extra spend metric: summed across all served cars.
        # Unlike _estimate_reward_for_bay (which prices a bay from its
        # congestion-free times, for policy decisions), this uses each
        # vehicle's REALISED cruise+walk time including queue waits.
        extra_total = 0.0
        served_count = 0
        for v in self.all_vehicles:
            if not v.assigned_bay:
                continue
            shop = _find_shop(self.carpark, v.destination_shop)
            if shop is None:
                continue
            actual_wasted = v.cruise_time_seconds + v.walk_time_seconds
            extra_wait_saved = max(0.0, REWARD_WORST_CASE_WASTED - actual_wasted)
            extra_shop_seconds = extra_wait_saved * REWARD_CONVERSION_RATE
            extra_spend = (extra_shop_seconds / 3600.0) * shop.spend_per_hour
            extra_total += extra_spend
            served_count += 1
        self.metrics.total_extra_spend_daily = extra_total
        self.metrics.avg_extra_spend_per_car = (
            extra_total / served_count if served_count else 0.0
        )

        self.metrics.throughput_per_hour = self.metrics.vehicles_served / 18.0
        self.metrics.max_concurrent_in_transit = self._max_concurrent


# ════════════════════════════════════════════════════════════════════════════
# Convenience runner
# ════════════════════════════════════════════════════════════════════════════

def run_comparison(carpark_builder, demand, policy_names=None,
                   day_type="Saturday", seed=42):
    if policy_names is None:
        policy_names = list(POLICIES.keys())
    results = {}
    for pname in policy_names:
        cp = carpark_builder()
        eng = SimulationEngine(cp, demand, pname, day_type, seed)
        eng.generate_arrivals()
        results[pname] = eng.run()
    return results
