"""
test_dqn_convergence.py

Tests DQN training stability across multiple random seeds by 
checking whether the reward curve stabilizes (does not diverge).

Note: Uses simulated reward curves as a placeholder until the full 
DQN training loop (Member 2, Day 4) produces real training logs.
"""

import numpy as np
from training_monitor import detect_divergence


def simulate_reward_curve(seed, num_episodes=500):
    """
    Placeholder simulation of a DQN reward curve for a given seed.
    Replace with real training logs once Member 2's training loop 
    (dqn_agent.py) is available.
    """
    np.random.seed(seed)
    # Simulated learning curve: reward increases and stabilizes with noise
    base_curve = np.linspace(0, 50, num_episodes)
    noise = np.random.normal(0, 5, num_episodes)
    return base_curve + noise


def test_convergence_across_seeds(seeds=(1, 2, 3)):
    results = {}
    for seed in seeds:
        reward_curve = simulate_reward_curve(seed)
        diverged = detect_divergence(list(reward_curve))
        results[seed] = {
            "final_avg_reward": np.mean(reward_curve[-50:]),
            "diverged": diverged
        }
        print(f"Seed {seed}: final_avg_reward={results[seed]['final_avg_reward']:.2f}, "
              f"diverged={diverged}")
    return results


if __name__ == "__main__":
    test_convergence_across_seeds()