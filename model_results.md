# Model Results Tracker

This document tracks all experiment results across baseline heuristics, 
tabular Q-Learning, and DQN training runs.

| Agent | Mean Revenue | Std Dev | Sell-through % | Notes |
|-------|-------------|---------|-----------------|-------|
| Random | - | - | - | Baseline, Week 1 |
| FixedPrice | - | - | - | Baseline, Week 2 |
| TimeBasedDiscount | - | - | - | Baseline, Week 2 |
| DemandBased | - | - | - | Baseline, Week 2 |
| Q-Learning | - | - | - | Tabular, Week 2 |
| DQN | - | - | - | Deep RL, Week 3 (in progress) |

## Week 3 Progress Notes (Day 1-2)

**Day 1**: Built PyTorch DQNNetwork architecture — 2 hidden layers with 
ReLU activation, output layer producing Q-values for each of the 10 
discrete price actions.

**Day 2**: Implemented target network with periodic hard update 
mechanism (`sync_target_network`). Verified weights correctly sync 
from policy network to target network on demand — this stabilizes 
DQN training by preventing the target Q-values from shifting every 
step (moving target problem).

**Q-Learning Baseline (for reference)**: Tabular Q-Learning agent from 
Week 2 serves as the primary comparison benchmark for DQN. Final 
cross-validated performance numbers pending from Member 3's evaluation.