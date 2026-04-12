"""
Demand Loader — Extract real arrival patterns from CSV/XLSX datasets
====================================================================
Reads the actual WPark datasets and builds probability distributions
for arrival rates, parking durations, and gate preferences.
The simulation replays these real patterns instead of using synthetic data.
"""

import os
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class DemandProfile:
    """Processed demand profile extracted from real data."""
    name: str
    # Hourly arrival rates: hour (0-23) → avg arrivals per hour
    hourly_arrival_rate: Dict[int, float] = field(default_factory=dict)
    # Weekday multipliers: day name → multiplier vs average
    weekday_multiplier: Dict[str, float] = field(default_factory=dict)
    # Monthly multipliers: month (1-12) → multiplier vs average
    monthly_multiplier: Dict[int, float] = field(default_factory=dict)
    # Duration distribution parameters (minutes)
    duration_mean: float = 120.0
    duration_std: float = 60.0
    duration_median: float = 117.0
    duration_percentiles: Dict[int, float] = field(default_factory=dict)
    # Raw duration samples for resampling
    duration_samples: Optional[np.ndarray] = None
    # Gate entry shares: gate_name → proportion
    gate_shares: Dict[str, float] = field(default_factory=dict)
    # Total stats
    total_entries: int = 0
    total_days: int = 0
    avg_daily_entries: float = 0.0

    def sample_duration(self, rng=None) -> float:
        """Sample a parking duration from the real distribution.
        Pass the engine's seeded RNG for deterministic results."""
        _rng = rng if rng is not None else np.random
        if self.duration_samples is not None and len(self.duration_samples) > 0:
            return float(_rng.choice(self.duration_samples))
        # Fallback: lognormal approximation
        return max(5.0, _rng.lognormal(
            np.log(self.duration_median), 0.6
        ))

    def get_arrival_rate(self, hour: int, weekday: str,
                         month: int = None) -> float:
        """Get expected arrivals for a specific hour/day/month."""
        base = self.hourly_arrival_rate.get(hour, 0.0)
        wd_mult = self.weekday_multiplier.get(weekday, 1.0)
        mo_mult = self.monthly_multiplier.get(month, 1.0) if month else 1.0
        return base * wd_mult * mo_mult


def build_synthetic_demand(name: str = "Synthetic Demo",
                           peak_arrivals_per_hour: int = 30) -> DemandProfile:
    """Build a clean, synthetic demand profile for the demo car park.
    Gives a smooth bell-curve arrival pattern peaking 11:00-14:00 and
    lognormal-ish duration distribution. Perfect for explaining the model
    without the noise of real-world data.
    """
    profile = DemandProfile(name=name)
    # Bell-curve arrival rates across the day (peak at 12:00)
    base_curve = {
        6: 0.05, 7: 0.10, 8: 0.25, 9: 0.45, 10: 0.70, 11: 0.90,
        12: 1.00, 13: 0.95, 14: 0.85, 15: 0.80, 16: 0.70, 17: 0.65,
        18: 0.60, 19: 0.50, 20: 0.35, 21: 0.20, 22: 0.10, 23: 0.03,
    }
    profile.hourly_arrival_rate = {
        h: rate * peak_arrivals_per_hour for h, rate in base_curve.items()
    }
    profile.weekday_multiplier = {
        "Monday": 0.75, "Tuesday": 0.75, "Wednesday": 0.80, "Thursday": 0.85,
        "Friday": 1.00, "Saturday": 1.25, "Sunday": 1.05,
        "Weekday Avg": 0.80,
    }
    # Duration samples: mix of short/medium/long visits
    rng = np.random.default_rng(42)
    short  = rng.normal(35, 10, 200).clip(10, 60)       # coffee / quick shop
    medium = rng.normal(85, 20, 300).clip(60, 120)      # fashion / restaurant
    long_  = rng.normal(150, 30, 200).clip(120, 240)    # cinema
    profile.duration_samples = np.concatenate([short, medium, long_])
    profile.duration_mean = float(np.mean(profile.duration_samples))
    profile.duration_median = float(np.median(profile.duration_samples))
    profile.duration_std = float(np.std(profile.duration_samples))
    profile.total_entries = sum(profile.hourly_arrival_rate.values()) * 7 * 52
    profile.total_days = 364
    profile.avg_daily_entries = sum(profile.hourly_arrival_rate.values())
    return profile


def load_all_datasets(data_dir: str) -> pd.DataFrame:
    """
    Load and concatenate all WPark datasets from the directory.
    Returns a single unified DataFrame.
    """
    file_map = [
        ("GAOct18-Mar19 (1).csv", "csv"),
        ("GAApr19-Sep19.xlsx", "xlsx"),
        ("GEOct18-Mar19.xlsx", "xlsx"),
        ("GEApr19-Sep19.xlsx", "xlsx"),
        ("GWDec18-Mar19.csv", "csv"),
        ("GWApr19-Sep19.xlsx", "xlsx"),
        ("PSOct18-Mar19.csv", "csv"),
        ("PSApr19-Sep19.xlsx", "xlsx"),
        ("QATOct18-Mar19 (3).csv", "csv"),
        ("QATApr19-Sep19.xlsx", "xlsx"),
    ]

    dfs = []
    for filename, fmt in file_map:
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            if fmt == "csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            # Drop any unnamed columns
            df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")

    if not dfs:
        raise FileNotFoundError(f"No datasets found in {data_dir}")

    combined = pd.concat(dfs, ignore_index=True)

    # Clean
    combined["Time"] = pd.to_datetime(combined["Time"], dayfirst=True,
                                       errors="coerce")
    combined["Parking Duration"] = pd.to_numeric(
        combined["Parking Duration"], errors="coerce"
    )

    return combined


def load_single_carpark(data_dir: str, prefix: str) -> pd.DataFrame:
    """Load data for a single car park (e.g. prefix='GA' for Grand Arcade)."""
    dfs = []
    for fname in os.listdir(data_dir):
        if fname.startswith(prefix) and (
            fname.endswith(".csv") or fname.endswith(".xlsx")
        ):
            path = os.path.join(data_dir, fname)
            if fname.endswith(".csv"):
                dfs.append(pd.read_csv(path))
            else:
                dfs.append(pd.read_excel(path))

    if not dfs:
        raise FileNotFoundError(f"No files with prefix '{prefix}' in {data_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.loc[:, ~combined.columns.str.startswith("Unnamed")]
    combined["Time"] = pd.to_datetime(combined["Time"], dayfirst=True,
                                       errors="coerce")
    combined["Parking Duration"] = pd.to_numeric(
        combined["Parking Duration"], errors="coerce"
    )
    return combined


def build_demand_profile(df: pd.DataFrame, name: str = "default") -> DemandProfile:
    """
    Build a DemandProfile from a cleaned DataFrame.
    Extracts real distributions for arrivals, durations, and gate usage.
    """
    profile = DemandProfile(name=name)

    # -- Time features --
    df = df.copy()
    df["hour"] = df["Time"].dt.hour
    df["weekday"] = df["Time"].dt.day_name()
    df["month"] = df["Time"].dt.month
    df["date"] = df["Time"].dt.date

    # -- Hourly arrival rate (avg entries per hour across all days) --
    daily_hourly = df.groupby(["date", "hour"])["Entries"].sum().reset_index()
    num_days = df["date"].nunique()
    hourly_totals = daily_hourly.groupby("hour")["Entries"].sum()
    for h in range(24):
        profile.hourly_arrival_rate[h] = hourly_totals.get(h, 0) / max(num_days, 1)

    # -- Weekday multiplier --
    weekday_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]
    weekday_totals = df.groupby("weekday")["Entries"].sum().reindex(weekday_order)
    weekday_days = df.groupby("weekday")["date"].nunique().reindex(weekday_order)
    weekday_avg = (weekday_totals / weekday_days).fillna(0)
    overall_daily_avg = weekday_avg.mean()
    if overall_daily_avg > 0:
        for day in weekday_order:
            profile.weekday_multiplier[day] = weekday_avg[day] / overall_daily_avg

    # -- Monthly multiplier --
    monthly_totals = df.groupby("month")["Entries"].sum()
    monthly_days = df.groupby("month")["date"].nunique()
    monthly_avg = (monthly_totals / monthly_days).fillna(0)
    overall_monthly_avg = monthly_avg.mean()
    if overall_monthly_avg > 0:
        for m in monthly_avg.index:
            profile.monthly_multiplier[int(m)] = monthly_avg[m] / overall_monthly_avg

    # -- Duration distribution --
    durations = df[df["Parking Duration"] > 0]["Parking Duration"].dropna()
    if len(durations) > 0:
        profile.duration_mean = durations.mean()
        profile.duration_std = durations.std()
        profile.duration_median = durations.median()
        profile.duration_percentiles = {
            25: durations.quantile(0.25),
            50: durations.quantile(0.50),
            75: durations.quantile(0.75),
            90: durations.quantile(0.90),
        }
        # Store up to 50k samples for resampling
        if len(durations) > 50000:
            profile.duration_samples = durations.sample(50000).values
        else:
            profile.duration_samples = durations.values

    # -- Gate shares --
    gate_entries = df.groupby("Device")["Entries"].sum()
    gate_entries = gate_entries[gate_entries > 0]
    total_gate = gate_entries.sum()
    if total_gate > 0:
        for gate, count in gate_entries.items():
            profile.gate_shares[gate] = count / total_gate

    # -- Totals --
    profile.total_entries = int(df["Entries"].sum())
    profile.total_days = num_days
    profile.avg_daily_entries = profile.total_entries / max(num_days, 1)

    return profile


def scale_demand_to_carpark(profile: DemandProfile,
                            target_capacity: int,
                            target_turnover: float = 3.0) -> DemandProfile:
    """
    Scale a demand profile to match a smaller/larger car park.
    target_turnover: expected number of full turnovers per day.
    E.g., 102 spaces × 3.0 turnovers = ~306 vehicles/day.
    """
    target_daily = target_capacity * target_turnover
    current_daily = profile.avg_daily_entries

    if current_daily == 0:
        return profile

    scale = target_daily / current_daily

    scaled = DemandProfile(
        name=f"{profile.name}_scaled",
        hourly_arrival_rate={
            h: rate * scale for h, rate in profile.hourly_arrival_rate.items()
        },
        weekday_multiplier=profile.weekday_multiplier.copy(),
        monthly_multiplier=profile.monthly_multiplier.copy(),
        duration_mean=profile.duration_mean,
        duration_std=profile.duration_std,
        duration_median=profile.duration_median,
        duration_percentiles=profile.duration_percentiles.copy(),
        duration_samples=profile.duration_samples,
        gate_shares=profile.gate_shares.copy(),
        total_entries=int(target_daily * profile.total_days),
        total_days=profile.total_days,
        avg_daily_entries=target_daily,
    )
    return scaled


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..")
    print("Loading Grand Arcade data...")
    df = load_single_carpark(data_dir, "GA")
    print(f"Loaded {len(df):,} rows")

    profile = build_demand_profile(df, "Grand Arcade")
    print(f"\nDemand Profile: {profile.name}")
    print(f"  Total entries:    {profile.total_entries:,}")
    print(f"  Avg daily:        {profile.avg_daily_entries:.0f}")
    print(f"  Duration median:  {profile.duration_median:.0f} min")
    print(f"  Peak hour rates:")
    for h in sorted(profile.hourly_arrival_rate,
                    key=profile.hourly_arrival_rate.get, reverse=True)[:5]:
        print(f"    {h:02d}:00 → {profile.hourly_arrival_rate[h]:.1f} arrivals/hr")

    # Scale to 102-space car park
    scaled = scale_demand_to_carpark(profile, 102, target_turnover=3.0)
    print(f"\nScaled to 102 spaces:")
    print(f"  Target daily: {scaled.avg_daily_entries:.0f}")
    for h in sorted(scaled.hourly_arrival_rate,
                    key=scaled.hourly_arrival_rate.get, reverse=True)[:5]:
        print(f"    {h:02d}:00 → {scaled.hourly_arrival_rate[h]:.1f} arrivals/hr")
