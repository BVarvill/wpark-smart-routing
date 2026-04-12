"""
rl_env.py — Gymnasium-compatible environment for RL training
=============================================================
Wraps the cellular car park simulator as a step-by-step environment
where each step is one arriving car.

    state  = [floor_occ_0, floor_occ_1, floor_occ_2,    # 3: per-floor %
              dest_floor_onehot,                          # 3: where customer wants
              stay_class_onehot,                          # 3: short/med/long
              hour_sin, hour_cos,                         # 2: time of day
              spend_rate_norm,                             # 1: shop spend / 50
              visit_min_norm,                              # 1: duration / 240
              congestion_norm]                             # 1: cars currently in transit
                                                          # Total: 14

    action = bay index (0..N_BAYS-1), masked to available bays

    reward = Option B: (time_saved × conversion × spend_rate / 3600) / visit_minutes
             Computed IMMEDIATELY per car (contextual bandit-style).

    done   = True when the day ends (all arrivals processed)

Usage:
    env = CarParkEnv(peak_rate=60)
    obs, info = env.reset(seed=42)
    while True:
        action = agent.select_action(obs, info["action_mask"])
        obs, reward, done, truncated, info = env.step(action)
        if done:
            break
"""
from __future__ import annotations
import math
import numpy as np
from typing import Optional, Dict, Any, Tuple

import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from carpark import (
    build_demo_carpark, CarPark, BayStatus,
    PARK_MANEUVER_SECONDS, STAIR_SECONDS_PER_FLOOR,
    WALK_SPEED_MPS, CAR_SPEED_MPS, METRES_PER_UNIT,
)
from demand import build_synthetic_demand, DemandProfile
from engine import (
    Vehicle, _find_shop, _estimate_reward_for_bay, total_walk_seconds,
)


# ── Constants ────────────────────────────────────────────────────────────
OBS_DIM = 25      # upgraded from 14 — richer congestion awareness
CONVERSION_RATE = 0.6
WORST_CASE_WASTED = 400.0


class CarParkEnv:
    """Gym-like (but no gym dependency) RL environment."""

    def __init__(self, peak_rate: int = 60, use_real_data: str = None,
                 data_dir: str = None):
        """
        peak_rate: synthetic peak arrivals/hour (ignored if use_real_data)
        use_real_data: e.g. "QAT" to use real arrival patterns
        data_dir: parent directory containing the CSVs
        """
        self.peak_rate = peak_rate
        self.use_real_data = use_real_data
        self.data_dir = data_dir or os.path.dirname(HERE)

        # Build a fresh carpark to get bay count
        cp = build_demo_carpark()
        self.n_bays = cp.total_capacity
        self.n_floors = len(cp.floors)

        # Spaces (for reference — no gym dependency)
        self.observation_space_shape = (OBS_DIM,)
        self.action_space_n = self.n_bays

    def reset(self, seed: int = 1) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset to a fresh day. Returns (observation, info)."""
        self.rng = np.random.RandomState(seed)
        self.carpark = build_demo_carpark()

        if self.use_real_data:
            from demand import load_single_carpark, build_demand_profile
            df = load_single_carpark(self.data_dir, self.use_real_data)
            self.demand = build_demand_profile(df, name=self.use_real_data)
            # Scale to fit our car park
            scale = self.n_bays / 780.0  # rough scale vs Grand Arcade
            for h in self.demand.hourly_arrival_rate:
                self.demand.hourly_arrival_rate[h] *= scale
        else:
            self.demand = build_synthetic_demand(
                "Demo", peak_arrivals_per_hour=self.peak_rate)

        # Generate arrivals for the day
        from engine import SimulationEngine
        eng = SimulationEngine(self.carpark, self.demand,
                               "nearest_entrance", "Saturday",
                               random_seed=int(self.rng.randint(1, 100000)))
        eng.generate_arrivals()
        self.arrivals = sorted(eng.arrival_schedule,
                               key=lambda v: v.arrival_second)
        self.arrival_idx = 0

        # Track bay status ourselves (simpler than running the full cellular sim)
        self.bay_occupied_until = {}  # bay_id → tick when it frees up
        self.current_tick = 0
        self.total_reward = 0.0
        self.cars_served = 0
        self.cars_rejected = 0

        # Metrics for evaluation
        self.episode_rewards = []
        self.episode_congestion = []

        # ── NEW: history tracking for richer state ──
        self.recent_arrivals = []       # timestamps of last N arrivals
        self.recent_assignments = []    # (floor, tick) of last N assignments
        self.per_floor_assigned_last_5min = [0, 0, 0]

        obs, info = self._get_next_arrival()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Assign the current car to bay[action].

        REWARD STRATEGY: day-level cumulative.
        Each step returns reward=0 EXCEPT the final step of the day,
        which returns the TOTAL daily revenue.  This forces the model
        to learn strategies that maximise the WHOLE DAY's outcome —
        not just one car at a time.  It's what enables lookahead:
        the model learns that a decision NOW affects revenue LATER.
        """
        if self.current_vehicle is None:
            return self._obs_zeros(), 0.0, True, False, {}

        v = self.current_vehicle

        # Get the chosen bay
        all_bays = [b for f in self.carpark.floors for b in f.bays]
        if action < 0 or action >= len(all_bays):
            self.cars_rejected += 1
            obs, info = self._get_next_arrival()
            done = self.arrival_idx >= len(self.arrivals) and self.current_vehicle is None
            # Penalty for invalid action (immediate — helps training stability)
            return obs, -0.01, done, False, info

        bay = all_bays[action]

        if bay.id in self.bay_occupied_until and self.bay_occupied_until[bay.id] > self.current_tick:
            self.cars_rejected += 1
            obs, info = self._get_next_arrival()
            done = self.arrival_idx >= len(self.arrivals) and self.current_vehicle is None
            return obs, -0.05, done, False, info

        # Assign the bay
        bay.status = BayStatus.OCCUPIED
        self.bay_occupied_until[bay.id] = (
            self.current_tick + int(v.duration_seconds) + 120
        )
        self.cars_served += 1

        # Track recent history for state features
        self.recent_arrivals.append(self.current_tick)
        self.recent_assignments.append((bay.floor, self.current_tick))
        # Keep only last 20
        self.recent_arrivals = self.recent_arrivals[-20:]
        self.recent_assignments = self.recent_assignments[-20:]
        # Per-floor assignments in last 5 min
        cutoff = self.current_tick - 300
        for fl in range(3):
            self.per_floor_assigned_last_5min[fl] = sum(
                1 for f, t in self.recent_assignments if f == fl and t > cutoff
            )

        # Accumulate per-car reward internally (NOT returned to agent)
        car_reward = _estimate_reward_for_bay(v, bay, self.carpark)
        self.total_reward += car_reward
        self.episode_rewards.append(car_reward)

        # Advance to next arrival
        obs, info = self._get_next_arrival()
        done = self.arrival_idx >= len(self.arrivals) and self.current_vehicle is None

        # PER-CAR REWARD: immediate feedback per assignment
        # (day-level was too sparse — +0.3% vs per-car's +2.7%)
        # The actor-critic's value function handles future estimation.
        return obs, car_reward, done, False, info

    def _get_next_arrival(self) -> Tuple[np.ndarray, Dict]:
        """Advance to the next pending arrival, freeing bays along the way."""
        while self.arrival_idx < len(self.arrivals):
            v = self.arrivals[self.arrival_idx]
            self.arrival_idx += 1
            self.current_tick = v.arrival_second

            # Free expired bays
            freed = [bid for bid, t in self.bay_occupied_until.items()
                     if t <= self.current_tick]
            for bid in freed:
                bay = self.carpark.get_bay(bid)
                if bay:
                    bay.status = BayStatus.AVAILABLE
                del self.bay_occupied_until[bid]

            self.current_vehicle = v
            obs = self._build_obs(v)
            mask = self._build_action_mask()

            if not any(mask):
                # No bays available — reject
                self.cars_rejected += 1
                continue

            return obs, {"action_mask": mask, "vehicle": v}

        self.current_vehicle = None
        return self._obs_zeros(), {"action_mask": np.zeros(self.n_bays, dtype=bool)}

    def _build_obs(self, v: Vehicle) -> np.ndarray:
        """Build the 25-dim observation vector.

        Upgraded from 14 dims to give the model congestion awareness:
          [0-2]   floor occupancy %                        (3) — how full each floor is
          [3-5]   per-floor recent pressure                (3) — cars assigned to each floor in last 5 min
          [6-8]   destination floor one-hot                (3) — where this customer wants to go
          [9-11]  stay class one-hot                       (3) — short / medium / long
          [12-13] time of day sin/cos                      (2) — cyclical time encoding
          [14]    spend rate normalised                    (1) — shop £/hr
          [15]    visit duration normalised                (1) — how long they'll stay
          [16]    total occupancy                          (1) — overall park fullness
          [17]    arrival rate (last 5 min)                (1) — demand surge detection
          [18]    arrival rate (last 15 min)               (1) — medium-term demand
          [19-21] per-floor available bays fraction        (3) — where there's space RIGHT NOW
          [22]    is peak hour                             (1) — binary: 11am-3pm
          [23]    cars rejected so far today               (1) — how stressed the system is
          [24]    time since last arrival                  (1) — gap between customers
        """
        # Floor occupancy % (3)
        occ = []
        for f in self.carpark.floors:
            n_occ = sum(1 for b in f.bays
                        if b.id in self.bay_occupied_until
                        and self.bay_occupied_until[b.id] > self.current_tick)
            occ.append(n_occ / max(1, f.capacity))

        # Per-floor recent pressure (3) — cars assigned in last 5 min
        pressure = [min(1.0, c / 10.0) for c in self.per_floor_assigned_last_5min]

        # Destination floor one-hot (3)
        dest = [0.0, 0.0, 0.0]
        dest[min(v.destination_floor, 2)] = 1.0

        # Stay class one-hot (3)
        dur_min = v.duration_seconds / 60.0
        if dur_min < 60:
            stay = [1.0, 0.0, 0.0]
        elif dur_min < 120:
            stay = [0.0, 1.0, 0.0]
        else:
            stay = [0.0, 0.0, 1.0]

        # Time features (2)
        hour = (v.arrival_second / 3600.0) % 24.0
        hour_sin = math.sin(2.0 * math.pi * hour / 24.0)
        hour_cos = math.cos(2.0 * math.pi * hour / 24.0)

        # Spend + duration (2)
        shop = _find_shop(self.carpark, v.destination_shop)
        spend_norm = min(1.0, (shop.spend_per_hour if shop else 0) / 50.0)
        visit_norm = min(1.0, v.duration_seconds / (240.0 * 60.0))

        # Total occupancy (1)
        total_occ = len(self.bay_occupied_until) / max(1, self.n_bays)

        # Arrival rate — last 5 min and 15 min (2)
        cutoff_5 = self.current_tick - 300
        cutoff_15 = self.current_tick - 900
        arrivals_5min = sum(1 for t in self.recent_arrivals if t > cutoff_5)
        arrivals_15min = sum(1 for t in self.recent_arrivals if t > cutoff_15)
        rate_5 = min(1.0, arrivals_5min / 15.0)    # normalise: 15 in 5min = very busy
        rate_15 = min(1.0, arrivals_15min / 40.0)

        # Per-floor available bays fraction (3)
        avail_per_floor = []
        for f in self.carpark.floors:
            n_avail = sum(1 for b in f.bays
                         if b.id not in self.bay_occupied_until
                         or self.bay_occupied_until[b.id] <= self.current_tick)
            avail_per_floor.append(n_avail / max(1, f.capacity))

        # Is peak hour (1)
        is_peak = 1.0 if 11 <= hour <= 15 else 0.0

        # Cars rejected so far (1)
        rejected_norm = min(1.0, self.cars_rejected / 100.0)

        # Time since last arrival (1)
        if len(self.recent_arrivals) >= 2:
            gap = self.current_tick - self.recent_arrivals[-2]
            gap_norm = min(1.0, gap / 300.0)  # normalise: 5min gap = max
        else:
            gap_norm = 1.0

        return np.array(
            occ                    # [0-2]   floor occupancy
            + pressure             # [3-5]   recent floor pressure
            + dest                 # [6-8]   destination one-hot
            + stay                 # [9-11]  stay class one-hot
            + [hour_sin, hour_cos] # [12-13] time
            + [spend_norm]         # [14]    spend rate
            + [visit_norm]         # [15]    visit duration
            + [total_occ]          # [16]    overall fullness
            + [rate_5, rate_15]    # [17-18] arrival rates
            + avail_per_floor      # [19-21] where there's space
            + [is_peak]            # [22]    peak indicator
            + [rejected_norm]      # [23]    system stress
            + [gap_norm],          # [24]    inter-arrival gap
            dtype=np.float32,
        )

    def _build_action_mask(self) -> np.ndarray:
        """Boolean mask: True for available bays."""
        all_bays = [b for f in self.carpark.floors for b in f.bays]
        mask = np.zeros(self.n_bays, dtype=bool)
        for i, b in enumerate(all_bays):
            if b.id not in self.bay_occupied_until or \
               self.bay_occupied_until[b.id] <= self.current_tick:
                mask[i] = True
        return mask

    def _obs_zeros(self) -> np.ndarray:
        return np.zeros(OBS_DIM, dtype=np.float32)


if __name__ == "__main__":
    # Quick self-test
    env = CarParkEnv(peak_rate=60)
    obs, info = env.reset(seed=1)
    print(f"Obs shape: {obs.shape}")
    print(f"Action mask sum: {info['action_mask'].sum()} available bays")
    print(f"N bays: {env.n_bays}")

    total_r = 0
    steps = 0
    while True:
        mask = info["action_mask"]
        if not any(mask):
            break
        # Random valid action
        valid = np.where(mask)[0]
        action = int(np.random.choice(valid))
        obs, reward, done, trunc, info = env.step(action)
        total_r += reward
        steps += 1
        if done:
            break

    print(f"Episode: {steps} steps, {env.cars_served} served, "
          f"{env.cars_rejected} rejected, total_reward={total_r:.2f}")
