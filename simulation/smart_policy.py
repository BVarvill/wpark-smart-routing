"""
smart_policy.py — Neural smart allocator
=========================================
A small PyTorch MLP that scores each arriving customer against the set
of currently-available parking bays, trained via imitation learning on
the greedy spend-rate policy.

Architecture
------------
Input features (22 dims total):
  Customer features (9 dims):
    - dest_floor one-hot [3]
    - stay_class one-hot [3] (short / medium / long)
    - visit_minutes_norm [1]  (duration / 240)
    - spend_per_hour_norm [1] (shop spend / 50)
    - hour_sin, hour_cos [2]

  Global state (13 dims):
    - floor occupancy %  [3]  (one per floor)
    - cars_in_transit_norm [1]
    - hour_of_day_norm [1]
    - concat of per-floor avg cruise time (unused here, future work) [3]
    - concat of per-floor avg walk time (unused here, future work) [3]
    - day_progress [1] (minute_of_day / 1080)
    - is_peak_hour [1]

Output: a 60-dim score vector (one score per bay in the 60-bay demo).
At inference time, scores for unavailable bays are masked to -∞ and
the argmax picks the bay.

Training
--------
Imitation learning on the greedy policy's chosen actions.  For each
arrival during simulation, record (state_features, greedy_bay_index)
and train the MLP with cross-entropy loss.

Usage
-----
    from smart_policy import NeuralSmartPolicy, select_bay_neural

    # Training (done offline, saves to models/smart_policy.pt)
    policy = NeuralSmartPolicy()
    policy.train_imitation(n_days=200)
    policy.save("models/smart_policy.pt")

    # Inference (called by engine.policy_neural_smart)
    bay_id = select_bay_neural(vehicle, carpark, available_bays)
"""
from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "smart_policy.pt")

# Feature dimensions
CUSTOMER_FEATS = 10   # 3 dest_floor + 3 stay_class + 1 visit_norm + 1 spend_norm + 2 hour(sin/cos)
GLOBAL_FEATS   = 13
INPUT_DIM      = CUSTOMER_FEATS + GLOBAL_FEATS  # 23
HIDDEN_DIM     = 64
# Output: one score per bay in the 60-bay demo
NUM_BAYS       = 60


# ═══════════════════════════════════════════════════════════════════════════
# Feature extraction (independent of torch so it can be tested on its own)
# ═══════════════════════════════════════════════════════════════════════════

def stay_class_onehot(duration_seconds: float) -> List[float]:
    """Short (<60min), Medium (60-120), Long (120+)."""
    m = duration_seconds / 60.0
    if m < 60:
        return [1.0, 0.0, 0.0]
    if m < 120:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def extract_customer_features(vehicle, shop) -> np.ndarray:
    """Return a 9-dim feature vector describing the arriving customer."""
    dest_floor = getattr(vehicle, "destination_floor", 0)
    dest_onehot = [0.0, 0.0, 0.0]
    dest_onehot[int(dest_floor)] = 1.0

    visit_sec = getattr(vehicle, "duration_seconds", 0.0) or 0.0
    visit_min_norm = min(1.0, visit_sec / (240.0 * 60.0))  # cap at 4h

    spend = 0.0
    if shop is not None:
        spend = getattr(shop, "spend_per_hour", 0.0) or 0.0
    spend_norm = min(1.0, spend / 50.0)

    arrival_sec = getattr(vehicle, "arrival_second", 0) or 0
    hour = (arrival_sec / 3600.0) % 24.0
    hour_sin = math.sin(2.0 * math.pi * hour / 24.0)
    hour_cos = math.cos(2.0 * math.pi * hour / 24.0)

    return np.array(
        dest_onehot
        + stay_class_onehot(visit_sec)
        + [visit_min_norm, spend_norm, hour_sin, hour_cos],
        dtype=np.float32,
    )


def extract_global_features(carpark, current_second: int = 0) -> np.ndarray:
    """Return a 13-dim feature vector describing the car park state."""
    floors = carpark.floors
    occ = [0.0, 0.0, 0.0]
    for i, f in enumerate(floors[:3]):
        occ[i] = f.occupancy_pct / 100.0

    # Cars in transit — best-effort from open cell occupancy
    in_transit = 0
    if hasattr(carpark, "lanes") and carpark.lanes:
        for lane in carpark.lanes.values():
            in_transit += sum(1 for c in lane.cells if c.car_id is not None)
    in_transit_norm = min(1.0, in_transit / 30.0)

    # Time of day features
    hour_norm = ((current_second or 0) / 86400.0)
    minute = (current_second or 0) / 60.0
    day_progress = min(1.0, minute / 1080.0)
    is_peak = 1.0 if 11 <= ((current_second or 0) / 3600.0) <= 15 else 0.0

    # Per-floor cruise/walk reserved for future work — zeros for now
    per_floor_cruise = [0.0, 0.0, 0.0]
    per_floor_walk   = [0.0, 0.0, 0.0]

    return np.array(
        occ
        + [in_transit_norm, hour_norm]
        + per_floor_cruise
        + per_floor_walk
        + [day_progress, is_peak],
        dtype=np.float32,
    )


def bay_index_of(bay) -> int:
    """Canonical bay index used by the neural network output.
    60-bay demo: bays are numbered F<floor>-<num> where num ∈ [1,20].
    Index = floor * 20 + (num - 1)."""
    return int(bay.floor) * 20 + (int(bay.number) - 1)


def bay_id_from_index(idx: int) -> str:
    floor = idx // 20
    num = (idx % 20) + 1
    return f"F{floor}-{num:03d}"


# ═══════════════════════════════════════════════════════════════════════════
# Network definition (if torch is available)
# ═══════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class SmartAllocator(nn.Module):
        """Small MLP: input (22) -> hidden (64) -> hidden (64) -> output (60).
        Each output dim is a score for one bay.  At inference time,
        unavailable bays are masked and the argmax is taken."""
        def __init__(self, input_dim: int = INPUT_DIM,
                     hidden_dim: int = HIDDEN_DIM,
                     num_bays: int = NUM_BAYS):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_bays),
            )

        def forward(self, x):
            return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════
# Inference helpers
# ═══════════════════════════════════════════════════════════════════════════

_CACHED_MODEL = None
_CACHED_MODEL_PATH = None


def _load_model_if_needed(path: str = MODEL_PATH):
    """Load the trained PyTorch model from disk on first use."""
    global _CACHED_MODEL, _CACHED_MODEL_PATH
    if not TORCH_AVAILABLE:
        return None
    if _CACHED_MODEL is not None and _CACHED_MODEL_PATH == path:
        return _CACHED_MODEL
    if not os.path.exists(path):
        return None
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model = SmartAllocator()
        model.load_state_dict(state_dict)
        model.eval()
        _CACHED_MODEL = model
        _CACHED_MODEL_PATH = path
        return model
    except Exception as e:
        print(f"[smart_policy] failed to load model: {e}")
        return None


def select_bay_neural(vehicle, carpark, available_bays) -> Optional[str]:
    """Score every bay with the trained model, mask to available bays,
    return the argmax bay id."""
    model = _load_model_if_needed()
    if model is None:
        return None   # caller will fall back to greedy

    # Build input features
    shop = None
    for f in carpark.floors:
        for s in f.shops:
            if s.name == getattr(vehicle, "destination_shop", ""):
                shop = s
                break
        if shop:
            break
    cust = extract_customer_features(vehicle, shop)
    glob = extract_global_features(carpark,
                                   getattr(vehicle, "arrival_second", 0))
    x = np.concatenate([cust, glob])
    x_t = torch.from_numpy(x).float().unsqueeze(0)

    with torch.no_grad():
        scores = model(x_t).squeeze(0).numpy()

    # Mask unavailable bays to -inf
    avail_idx = {bay_index_of(b): b.id for b in available_bays}
    masked = np.full(NUM_BAYS, -1e9, dtype=np.float32)
    for idx, bid in avail_idx.items():
        if 0 <= idx < NUM_BAYS:
            masked[idx] = scores[idx]
    best_idx = int(np.argmax(masked))
    return avail_idx.get(best_idx)


# ═══════════════════════════════════════════════════════════════════════════
# Training (imitation learning from greedy)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingExample:
    features: np.ndarray       # shape (INPUT_DIM,)
    reward_vector: np.ndarray  # shape (NUM_BAYS,) — per-bay Option B reward


def collect_imitation_data(n_days: int = 50, peak_rate: int = 60,
                           random_seed_base: int = 1000,
                           verbose: bool = True) -> List[TrainingExample]:
    """Run the greedy policy for N simulated days.  At each arrival,
    capture the state features and the FULL per-bay reward vector
    (not just the greedy's argmax).  This turns the training problem
    into a regression that doesn't depend on which bays were available
    at any given moment."""
    from carpark import build_demo_carpark, BayStatus
    from demand import build_synthetic_demand
    from engine import (
        SimulationEngine, policy_greedy_smart, _estimate_reward_for_bay,
        _find_shop,
    )

    examples: List[TrainingExample] = []

    for day in range(n_days):
        seed = random_seed_base + day
        cp = build_demo_carpark()
        demand = build_synthetic_demand("Demo", peak_arrivals_per_hour=peak_rate)
        eng = SimulationEngine(cp, demand, "greedy_smart", "Saturday",
                               random_seed=seed)
        eng.generate_arrivals()

        # Reset carpark state
        for f in cp.floors:
            for b in f.bays:
                b.status = BayStatus.AVAILABLE
                b.occupied_by = None

        # Bay release times (approximation of turnover)
        all_bays = [b for f in cp.floors for b in f.bays]
        bay_free_at = {b.id: 0 for b in all_bays}

        arrivals = sorted(eng.arrival_schedule, key=lambda x: x.arrival_second)

        for v in arrivals:
            # Release bays whose time has come
            for bid, t in list(bay_free_at.items()):
                if t > 0 and t <= v.arrival_second:
                    bay_obj = cp.get_bay(bid)
                    if bay_obj is not None:
                        bay_obj.status = BayStatus.AVAILABLE
                        bay_obj.occupied_by = None
                    bay_free_at[bid] = 0

            available = [b for b in all_bays if b.is_available()]
            if not available:
                continue

            shop = _find_shop(cp, v.destination_shop)
            cust = extract_customer_features(v, shop)
            glob = extract_global_features(cp, v.arrival_second)
            features = np.concatenate([cust, glob])

            # Compute reward for every bay (available or not — the
            # network learns the reward function, masking happens at
            # inference time).
            reward_vector = np.zeros(NUM_BAYS, dtype=np.float32)
            for bay in all_bays:
                idx = bay_index_of(bay)
                if 0 <= idx < NUM_BAYS:
                    reward_vector[idx] = _estimate_reward_for_bay(v, bay, cp)

            examples.append(TrainingExample(
                features=features,
                reward_vector=reward_vector,
            ))

            # Apply greedy's choice so the next arrival sees updated state
            bay_id = policy_greedy_smart(v, cp)
            if bay_id is None:
                continue
            bay = cp.get_bay(bay_id)
            if bay is None:
                continue
            bay.status = BayStatus.OCCUPIED
            bay.occupied_by = v.id
            bay_free_at[bay_id] = (v.arrival_second
                                   + int(v.duration_seconds)
                                   + 120)

        if verbose and (day + 1) % 10 == 0:
            print(f"  Collected {len(examples)} examples from {day+1}/{n_days} days",
                  flush=True)

    if verbose:
        print(f"  Total training examples: {len(examples)}")
    return examples


def train_neural_policy(examples: List[TrainingExample],
                        n_epochs: int = 40,
                        batch_size: int = 256,
                        lr: float = 1e-3,
                        verbose: bool = True) -> Tuple["SmartAllocator", List[float]]:
    """Train the MLP via regression against the per-bay reward vectors.
    Loss = MSE.  At inference, scores are masked to available bays and
    the argmax is taken — a.k.a. the greedy rule applied to predictions."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed")

    X = np.array([ex.features for ex in examples], dtype=np.float32)
    Y = np.array([ex.reward_vector for ex in examples], dtype=np.float32)

    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)

    model = SmartAllocator()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = len(examples)
    losses = []

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        X_ep = X_t[perm]
        Y_ep = Y_t[perm]

        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            xb = X_ep[i:i+batch_size]
            yb = Y_ep[i:i+batch_size]
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(1, n_batches)
        losses.append(avg_loss)

        if verbose and (epoch % 5 == 0 or epoch == n_epochs - 1):
            # Measure argmax agreement with the greedy choice
            with torch.no_grad():
                preds_full = model(X_t)
                greedy_argmax = Y_t.argmax(dim=1)
                pred_argmax   = preds_full.argmax(dim=1)
                agreement = (greedy_argmax == pred_argmax).float().mean().item()
                # Also compute rank correlation-ish metric: how often is
                # the greedy's choice in the network's top-5
                top5 = preds_full.topk(5, dim=1).indices
                in_top5 = (top5 == greedy_argmax.unsqueeze(1)).any(dim=1).float().mean().item()
            print(f"  Epoch {epoch+1:3d}/{n_epochs}  loss={avg_loss:.6f}  "
                  f"argmax_match={agreement*100:.1f}%  top5={in_top5*100:.1f}%",
                  flush=True)

    return model, losses


def save_model(model, path: str = MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  Saved model to {path}")


if __name__ == "__main__":
    # Quick self-test
    print("smart_policy.py — self test")
    print(f"  torch available: {TORCH_AVAILABLE}")
    print(f"  INPUT_DIM = {INPUT_DIM}")
    print(f"  NUM_BAYS = {NUM_BAYS}")
    if TORCH_AVAILABLE:
        m = SmartAllocator()
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  Model parameters: {n_params:,}")
