# Results Comparison

This document tracks the performance comparison of all pricing agents 
across the project sprints.

## 1. Baseline Comparison
*(To be filled in Week 2 — Random, Fixed, Time-Based Discount, Demand-Based agents)*

## 2. Q-Learning Results
## Agent Comparison Table (In Progress)

| Agent | Mean Revenue | Std Dev | Sell-Through Rate | Status |
|-------|-------------|---------|---------------------|--------|
| Random | - | - | - | ✅ Complete (Week 1) |
| Fixed | - | - | - | ✅ Complete (Week 2) |
| Time-Based Discount | - | - | - | ✅ Complete (Week 2) |
| Demand-Based | - | - | - | ✅ Complete (Week 2) |
| Q-Learning | - | - | - | ✅ Complete (Week 2) |
| DQN | - | - | - | 🚧 In Progress (Week 3) |

*Note: Numerical values to be filled in once Member 3's full evaluation 
pass is complete for all agents.*

## 3. DQN Results
*(To be filled in Week 3 — Deep Q-Network performance vs Q-Learning and baselines)*

## 4. Final Policy Recommendation
*(To be filled in Week 4 — business recommendations for deployment)*
## Week 2 Findings Summary

This week, three heuristic baseline pricing agents were implemented and 
evaluated: FixedPrice (constant price throughout the season), 
TimeBasedDiscount (price decays 10% daily as the deadline approaches), 
and DemandBased (price adjusts dynamically based on the ratio of 
remaining inventory to remaining time).

Each agent was tested across 500 simulated booking seasons. Results show 
that heuristic strategies which adapt to time or demand generally 
outperform a static fixed price, since real booking behavior is highly 
time-sensitive — customers are more price-sensitive early in the season 
and less sensitive closer to departure.

In parallel, a tabular Q-Learning agent was implemented and trained on 
a discretized state space (inventory buckets, days buckets) using the 
Bellman update rule with an epsilon-greedy exploration strategy. Unlike 
the heuristics, which follow a fixed, hand-crafted rule regardless of 
outcome, Q-Learning learns an optimal pricing policy directly from 
simulated experience — allowing it to capture more nuanced patterns in 
the interaction between inventory, time, and price than any single 
heuristic rule can.

Early results indicate Q-Learning outperforms the best-performing 
heuristic baseline in mean episodic revenue, validating the case for 
moving toward learned policies in Week 3's Deep Q-Network. Full 
statistical comparison to follow once Member 3's evaluation is complete.


## Conclusion (Week 3)

Across all tested strategies, agents that adapt their pricing decisions 
based on state (inventory and time remaining) consistently outperform 
static approaches. Among the adaptive strategies, the learning-based 
agents (Q-Learning and DQN) are expected to outperform hand-crafted 
heuristics, since they optimize directly for cumulative revenue rather 
than following a fixed rule.

DQN is expected to outperform tabular Q-Learning specifically because 
the pricing problem has a state space that, while represented here with 
just two features, benefits from the function approximation a neural 
network provides — allowing smoother generalization across unseen 
inventory/time combinations rather than requiring every state to be 
visited and discretized individually, as tabular Q-Learning requires.

Final numerical comparison (mean revenue, std dev, sell-through rate) 
will be completed once Member 2's DQN training run (Week 3, Day 5) and 
Member 3's full evaluation pass are finalized.