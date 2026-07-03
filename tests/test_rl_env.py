"""Contract tests for the RL environment."""
import numpy as np

from rl_env import CarParkEnv, OBS_DIM


def test_obs_shape_and_mask():
    env = CarParkEnv(peak_rate=40)
    obs, info = env.reset(seed=3)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    mask = info["action_mask"]
    assert mask.shape == (env.n_bays,)
    assert mask.all(), "empty car park should have every bay available"


def test_obs_within_observation_space_bounds():
    """train_ppo declares Box(0, 1.5); every emitted obs must respect it
    (hour_sin/cos can be negative — the Box low bound would clip them,
    so this test documents the real range)."""
    env = CarParkEnv(peak_rate=40)
    obs, info = env.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(200):
        mask = info.get("action_mask")
        if mask is None or not mask.any():
            break
        action = int(rng.choice(np.where(mask)[0]))
        obs, reward, done, trunc, info = env.step(action)
        assert obs.max() <= 1.5 + 1e-6
        assert obs.min() >= -1.0 - 1e-6   # sin/cos floor
        if done:
            break


def test_full_episode_serves_cars():
    env = CarParkEnv(peak_rate=40)
    obs, info = env.reset(seed=5)
    rng = np.random.default_rng(1)
    total_reward = 0.0
    while True:
        mask = info.get("action_mask")
        if mask is None or not mask.any():
            break
        action = int(rng.choice(np.where(mask)[0]))
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        if done:
            break
    assert env.cars_served > 0
    assert total_reward > 0.0


def test_assignment_books_the_bay():
    env = CarParkEnv(peak_rate=40)
    obs, info = env.reset(seed=7)
    v = info["vehicle"]
    first_action = int(np.where(info["action_mask"])[0][0])
    all_bays = [b for f in env.carpark.floors for b in f.bays]
    bay_id = all_bays[first_action].id
    obs, reward, done, trunc, info = env.step(first_action)
    # The bay is booked until the customer's departure (plus exit buffer)
    assert env.bay_occupied_until[bay_id] >= \
        v.arrival_second + int(v.duration_seconds)
    assert reward > 0.0
