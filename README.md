# WPark Smart Car Park Routing

Physics simulation of a multi-storey car park with reinforcement learning for revenue-optimised parking allocation. Built during a 4 day sprint at Cambridge Judge Business School with WPark (backed by Google, Nvidia, Microsoft).

## What this is

67% of shoppers in multi-storey car parks end up on the wrong floor for their desired shop. Every wrong floor costs 45 seconds of stair climbing. At £22/hour average spend, that friction costs the mall real money.

I built a simulator that models this problem physically (cellular automaton, every second tracked) and tested three routing policies across 1,000 simulated days:

**Baseline:** park in the first available space. Ground floor fills first. No awareness of which shop the customer wants.

**Floor Match:** if we know which shop they want, send them to that floor. Saves 15 seconds per customer. Generates £17k/year extra revenue.

**PPO (reinforcement learning):** a neural network trained from scratch to maximise daily revenue. No human designed rules. It discovered its own strategy, saves 21 seconds per customer, and generates £24k/year. Outperforms the hand coded rule by 6.7%.

## How it works

The simulator is a cellular automaton. Cars occupy discrete cells on a one way lane network across 3 floors. Each cell holds one car. If the cell ahead is occupied, the car waits. Parking blocks the lane for 45 seconds while the car reverses in. Congestion emerges from the physics, not from a formula.

The RL model uses a 25 dimensional state (floor occupancy, arrival rates, congestion indicators, customer features) and picks from 60 possible bays with invalid actions masked. Trained via MaskablePPO from stable-baselines3 for 1M+ timesteps on CPU in about 15 minutes.

## Running it

```bash
cd simulation

# live simulation
python demo_pygame.py

# compare all policies
python compare_policies.py --verbose

# train RL from scratch
python train_ppo.py --steps 1000000

# web dashboard
streamlit run webapp.py
```

Pygame controls: 1 baseline, 2 floor match, 3 RL policy, space to pause, arrows for speed.

## Results

All validated across 1,000 simulations, p < 0.001 on every comparison.

| Policy | Time saved per car | Extra revenue per year | Congestion |
|---|---|---|---|
| Baseline | n/a | n/a | 6.5s |
| Floor Match | 15s | £17,095 | 5.7s |
| PPO (RL) | 21s | £24,454 | 3.4s |

The 60 bay demo projects to roughly £892k/year for a full scale shopping centre.

## What the RL model learned

It figured out that not all seconds are equal. A grocery customer with 30 minutes to shop benefits more from a premium bay than a cinema customer with 2.5 hours. So it gives short stay high spend customers the closest bays and routes long stay customers slightly further. Average time saved barely changes but the revenue value of those seconds goes up because they are targeted where they matter most.

## Files

```
simulation/
  carpark.py            car park geometry and cellular lane model
  engine.py             simulation engine and routing policies
  demand.py             demand generation from real Cambridge data
  rl_env.py             gymnasium compatible RL environment
  train_ppo.py          PPO training script
  demo_pygame.py        live pygame visualisation
  webapp.py             streamlit dashboard
  compare_policies.py   side by side policy comparison
  run_baseline_study.py 1000 run statistical study
  sim_results.py        single source of truth for metrics
  smart_policy.py       neural network policy and imitation learning
  models/
    ppo_policy.zip      trained PPO model (1M steps)
results/
  combined_4policy_1000.csv  full study results
```

## Data

Simulated demand is calibrated against real Cambridge car park data (3.6M vehicle events across 5 car parks). Shop destinations are modelled synthetically based on peak hour weighting.
