"""
train_ppo.py — PPO training via stable-baselines3
===================================================
Uses the industry-standard PPO implementation with our 25-dim state
car park environment. No warm-start — learns from scratch.

PPO advantages over REINFORCE:
  - Clipped objective prevents catastrophic policy updates
  - Value function baseline reduces gradient variance
  - Multiple epochs per rollout = more sample-efficient
  - Parallel environments for faster data collection

Usage:
    python train_ppo.py                        # 50k steps (~5 min, sanity check)
    python train_ppo.py --steps 5000000        # full training (~75 min, the shipped model)
    python train_ppo.py --peak 80              # busier car park
"""
import argparse, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from rl_env import CarParkEnv, OBS_DIM
from engine import policy_greedy_smart


class CarParkGymEnv(gym.Env):
    """Gymnasium wrapper with action masking for MaskablePPO."""

    metadata = {"render_modes": []}

    def __init__(self, peak_rate=60):
        super().__init__()
        self.inner = CarParkEnv(peak_rate=peak_rate)
        self.observation_space = spaces.Box(
            low=0.0, high=1.5, shape=(OBS_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.inner.n_bays)
        self._seed_counter = 0
        self._mask = np.ones(self.inner.n_bays, dtype=bool)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed_counter = seed
        else:
            self._seed_counter += 1
        obs, info = self.inner.reset(seed=self._seed_counter)
        self._mask = info.get("action_mask", np.ones(self.inner.n_bays, dtype=bool))
        return obs, info

    def step(self, action):
        action = int(action)
        obs, reward, done, truncated, info = self.inner.step(action)
        self._mask = info.get("action_mask", np.ones(self.inner.n_bays, dtype=bool))
        return obs, float(reward), bool(done), bool(truncated), info

    def action_masks(self) -> np.ndarray:
        """Called by MaskablePPO to get the current valid action mask."""
        return self._mask


class ProgressCallback(BaseCallback):
    """Print progress during training."""
    def __init__(self, eval_env, eval_every=5000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_every = eval_every
        self.best_reward = -float("inf")

    def _on_step(self):
        if self.num_timesteps % self.eval_every == 0:
            rewards = []
            for seed in range(1, 11):
                obs, info = self.eval_env.reset(seed=seed + 90000)
                total = 0.0
                done = False
                while not done:
                    mask = self.eval_env.action_masks()
                    action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
                    obs, reward, done, trunc, info = self.eval_env.step(action)
                    total += reward
                rewards.append(total)
            avg = np.mean(rewards)
            if avg > self.best_reward:
                self.best_reward = avg
            print(f"  step {self.num_timesteps:6d}  eval_reward={avg:.3f}  "
                  f"best={self.best_reward:.3f}", flush=True)
        return True


def evaluate_final(model, env, n=50, seed_offset=80000):
    rewards = []
    served = []
    for i in range(n):
        obs, info = env.reset(seed=seed_offset + i)
        total = 0.0
        done = False
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, reward, done, trunc, info = env.step(action)
            total += reward
        rewards.append(total)
        served.append(env.inner.cars_served)
    return np.mean(rewards), np.std(rewards), np.mean(served)


def evaluate_greedy(env, n=50, seed_offset=80000):
    """Evaluate the hand-coded greedy policy for comparison."""
    rewards = []
    for i in range(n):
        obs, info = env.reset(seed=seed_offset + i)
        total = 0.0
        done = False
        while not done:
            v = info.get("vehicle")
            if v is None:
                break
            bay_id = policy_greedy_smart(v, env.inner.carpark)
            if bay_id is None:
                break
            all_bays = [b for f in env.inner.carpark.floors for b in f.bays]
            action = next((j for j, b in enumerate(all_bays) if b.id == bay_id), 0)
            obs, reward, done, trunc, info = env.step(action)
            total += reward
        rewards.append(total)
    return np.mean(rewards), np.std(rewards)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50000,
                        help="total training timesteps (default 50k)")
    parser.add_argument("--peak", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", type=str, default="models/ppo_policy.zip",
                    help="output path — engine.py loads models/ppo_policy.zip")
    args = parser.parse_args()

    print("=" * 64)
    print("  WPark — PPO Training (stable-baselines3)")
    print("=" * 64)
    print(f"  Total steps:  {args.steps:,}")
    print(f"  Peak rate:    {args.peak}/hr")
    print(f"  State dim:    {OBS_DIM}")
    print(f"  Action space: {60} bays")
    print()

    # Create environments
    train_env = CarParkGymEnv(peak_rate=args.peak)
    eval_env = CarParkGymEnv(peak_rate=args.peak)

    # Evaluate greedy baseline first
    print("  [baseline] Evaluating greedy policy...", flush=True)
    greedy_avg, greedy_std = evaluate_greedy(eval_env, n=20)
    print(f"  [baseline] Greedy avg_reward = {greedy_avg:.3f} ± {greedy_std:.3f}")
    print()

    # Build MaskablePPO — PPO with proper action masking
    # The model ONLY considers valid (unoccupied) bays, never wastes
    # exploration on invalid actions.
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.lr,
        n_steps=1024,           # rollout buffer size
        batch_size=64,
        n_epochs=10,            # PPO epochs per rollout
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # exploration bonus
        verbose=0,
        policy_kwargs=dict(
            net_arch=[128, 128],
        ),
    )

    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  PPO model: {n_params:,} parameters")
    print()

    # Evaluate before training
    print("  [eval] Before training:", flush=True)
    pre_avg, pre_std, pre_served = evaluate_final(model, eval_env, n=20)
    print(f"    avg_reward={pre_avg:.3f} ± {pre_std:.3f}  served={pre_served:.0f}")
    print()

    # Train
    print("  Training PPO...", flush=True)
    t0 = time.time()
    callback = ProgressCallback(eval_env, eval_every=5000)
    model.learn(total_timesteps=args.steps, callback=callback)
    train_time = time.time() - t0
    print(f"\n  Training complete in {train_time:.0f}s")
    print()

    # Final evaluation
    print("  [eval] After training:", flush=True)
    post_avg, post_std, post_served = evaluate_final(model, eval_env, n=50)
    print(f"    avg_reward={post_avg:.3f} ± {post_std:.3f}  served={post_served:.0f}")

    improvement_vs_random = (post_avg - pre_avg) / max(abs(pre_avg), 0.001) * 100
    improvement_vs_greedy = (post_avg - greedy_avg) / max(abs(greedy_avg), 0.001) * 100
    print(f"\n  vs untrained:  {improvement_vs_random:+.1f}%")
    print(f"  vs greedy:     {improvement_vs_greedy:+.1f}%")

    # Save
    out_path = os.path.join(HERE, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.save(out_path)
    print(f"  Model saved: {out_path}")

    # Log
    with open(os.path.join(HERE, "ppo_training_log.txt"), "w") as f:
        f.write(f"PPO Training Log\n")
        f.write(f"================\n")
        f.write(f"Steps: {args.steps}\n")
        f.write(f"Training time: {train_time:.0f}s\n")
        f.write(f"Pre-training: {pre_avg:.4f}\n")
        f.write(f"Post-training: {post_avg:.4f}\n")
        f.write(f"Greedy baseline: {greedy_avg:.4f}\n")
        f.write(f"vs untrained: {improvement_vs_random:+.1f}%\n")
        f.write(f"vs greedy: {improvement_vs_greedy:+.1f}%\n")


if __name__ == "__main__":
    main()
