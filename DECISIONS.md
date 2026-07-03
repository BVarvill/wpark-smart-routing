# Design decisions

This file records the significant design choices in this project, the alternatives considered, and why each choice won.
It exists because a simulation whose numbers drive a business case should be able to defend every assumption out loud.

## 1. Congestion via cellular automaton, not queueing formulas

**Decision:** model every lane as a chain of discrete cells, one car per cell, advancing only into empty cells.
A parking car holds its approach cell for the full 45-second manoeuvre, physically blocking the lane behind it.

**Alternatives rejected:**
- A tuned congestion multiplier (`travel_time × f(occupancy)`).
  Rejected because the multiplier is a free parameter you can bend to any conclusion, and an interviewer should ask "where did that curve come from?"
- M/M/c-style queueing theory.
  Rejected because car park lanes are not memoryless service queues; the dominant effect is spatial blocking (you are stuck behind the exact car parking ahead of you), which queueing formulas abstract away.

**Why it matters:** congestion is an emergent output, not an input.
No-overtaking is true by construction, so the "stuck behind a parking car" effect the business case leans on cannot be an artefact of tuning.
The trade-off is cost: the pre-pass simulates every second of the day, which is why the metric loop replays pre-computed timelines instead of re-simulating.

## 2. Two-phase engine: second-level physics pre-pass, minute-level metric replay

**Decision:** run the cellular simulation once at 1-second resolution, stamp each vehicle with its realised timeline, then compute metrics and UI frames from those timelines at minute resolution.

**Alternative rejected:** a single loop doing both.
Physics needs 1-second ticks (a 45-second manoeuvre at minute resolution rounds to 1 tick and loses the blocking effect); metrics and rendering do not.
Doing both at 1 second makes every UI scrub and metric query 60x more expensive for no accuracy gain.

## 3. Reward design: "Option B" revenue per minute of visit

**Decision:** the value of assigning a car to a bay is
`max(0, 400s worst case - (cruise + walk)) × 0.6 conversion × shop £/hr ÷ visit minutes`.

**The interesting term is the division by visit minutes.**
A 30-minute grocery shopper loses a much larger fraction of their visit to a 60-second walk than a 150-minute cinema-goer does, so a second saved is worth more revenue for the short-stay customer.
This single term is what the PPO agent later exploits: it gives premium bays to short-stay, high-spend customers.

**Alternatives rejected:**
- Minimise average wasted time (treats all seconds as equal; provably leaves revenue on the table given heterogeneous spend rates).
- Maximise raw spend per customer without the visit-minutes normalisation (over-rewards long-stay customers who barely notice the saving).

**Calibration:** the 0.6 dwell-to-spend conversion comes from published UK mall research (Dennis et al. 2002; Underhill 1999 report a 0.5 to 0.85 range) and is deliberately the conservative end.
The 400s worst case is the slowest realistic path through the demo park.
Both constants live in one place (`engine.py` reward constants, cited in `sim_results.py`) and every downstream number inherits them.

**An assumption worth flagging separately from the conversion rate itself:** citing 0.6 justifies the *rate*, not the underlying mechanism.
The whole reward function assumes a customer who saves time on parking and walking spends that time shopping instead of leaving earlier.
That is plausible and broadly supported by dwell-time literature in general, but neither this simulation nor the source dataset (parking events, not till receipts) can confirm it holds for this specific car park.
The honest position is: 0.6 is a defensible number to use, not proof the mechanism is real here - a pilot with point-of-sale data is what would settle it.

## 4. Greedy = the analytic optimum of the reward, used as the bar RL must clear

**Decision:** implement `policy_greedy_smart` as the exact per-car argmax of the reward function, and treat it as the strongest fair baseline.

**Why:** if you own the reward function, the per-decision optimum is computable, so a learned policy that "beats a baseline" has proven nothing unless the baseline IS that optimum.
Greedy can only be beaten through inter-car strategy (sacrificing this car's best bay to protect a later, more valuable assignment).
That framing makes the RL result interpretable: any margin over greedy is evidence of learned lookahead, not of a weak baseline.

## 5. MaskablePPO over vanilla PPO, DQN, or REINFORCE

**Decision:** sb3-contrib MaskablePPO, MLP policy, 25-dim state, 60-way discrete action with invalid actions masked.

**Why masking:** at high occupancy most of the 60 actions are invalid.
Without masking the agent wastes most of its exploration learning "don't pick occupied bays", which the environment already knows.
**Why PPO over DQN:** a stochastic clipped-objective policy-gradient method is robust to the non-stationarity here (the bay-availability distribution shifts through the day).
**Why not REINFORCE:** tried first in the sprint; the variance was unusable, which is what pushed the project to PPO.

## 6. Per-car reward, not day-level sparse reward

**Decision:** each step returns that assignment's revenue value immediately.

**Alternative tried and rejected:** return 0 every step and total daily revenue on the last step, to force whole-day optimisation.
It trained far worse (+0.3% over greedy vs +2.7% for per-car in sprint measurements).
With roughly 500 assignments per episode, credit assignment across a day was too hard, and PPO's value function already estimates future consequences of each assignment.

## 7. Training environment approximates the car park without congestion

**Decision (and known limitation):** `rl_env.py` tracks bay occupancy with a simple booking table and computes rewards from congestion-free times.
The full cellular simulation is only used at evaluation time.

**Why:** training throughput.
Cheap steps are what make 5M steps in ~75 minutes on a CPU possible; stepping the cellular simulator inside the training loop would be orders of magnitude slower.

**The honest cost:** the agent cannot learn congestion avoidance from a reward signal that contains no congestion, and evaluation happens in the congested simulator.
Related: at inference time inside the engine, a few state dimensions (recent-assignment pressure, arrival rates, rejection count) are occupancy-derived proxies for histories the engine does not track, so there is train/serve skew on those dims.
These two facts are the leading explanation for why PPO's margin over greedy is small.
Fixing it (training inside the cellular sim, wiring real histories through the engine) is the top item on the roadmap and would be the first thing to do with more time.

## 8. Synthetic demand in the repo, real data kept private

**Decision:** every committed entry point uses a hand-authored synthetic demand profile - a bell-curve arrival-rate curve and three normal-distributed stay-length clusters (short/medium/long) - rather than the real Cambridge dataset.

**Precision about what "real data" contributed:** the synthetic profile's general shape (arrivals peaking around midday, a mix of short/medium/long stays) is the kind of pattern real car parks show, but the specific numbers were typed by hand for a clean, explainable demo - they were not fitted to the real Cambridge dataset (3.6M vehicle events across 5 car parks).
`demand.py` includes loaders (`load_single_carpark`, `build_demand_profile`) capable of extracting a real arrival-rate curve AND a real stay-length distribution from that dataset - both dimensions, not only arrivals - but the raw files are WPark's and are not distributed, so no committed script uses them by default.

**Consequence to state plainly:** every number in the README is a property of the simulator under synthetic demand, not a measured property of a real car park.
The "68% wrong floor" figure is the simulator's own baseline output, not an industry statistic.

## 9. Statistics: 1000 paired runs, hand-rolled normal-approximation t-test

**Decision:** compare policies on 1000 simulated days per policy, with the same seed per day across policies (paired design), and report mean, 95% CI, and a paired t-test.

**Why paired:** policies face identical customer streams, so day-level demand noise cancels in the differences and the test gains power.
**Why the t-test is hand-rolled:** the repo needs no scipy for anything else, and at N=1000 the normal approximation to the t distribution is exact to more decimal places than we report.
Pulling in a dependency to replace 20 lines of Abramowitz & Stegun was judged worse for a repo whose value is transparency.

## 10. Loud failure over graceful degradation in the PPO loader

**Decision:** if the PPO model cannot load, the engine logs an error and falls back to greedy; nothing is caught silently.

**History:** an earlier version looked for a differently-named model file, silently fell back to greedy, and labelled the results "PPO".
The bug produced a study in which "PPO" and greedy were statistically identical, and it survived because the fallback printed one easily-missed line.
The regression test `test_ppo_differs_from_greedy` now fails the suite if the model is missing or inert, and the loader treats a missing model as an error-level event.

## 11. Three geometries, one retained for provenance

`carpark.py` builds three layouts: the 60-bay cellular demo (default everywhere), a parametric scaled variant (120 to 180 bays, same physics), and "Car Park A", a 447-bay digitisation of the real architectural drawing that uses an older segment-FIFO congestion model.
Car Park A is retained because it documents the project's origin against a real building plan, but the cellular demo is the model all shipped numbers come from.
If this were production code rather than a portfolio of the sprint's work, Car Park A and its legacy engine path would be deleted.
