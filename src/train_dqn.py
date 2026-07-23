"""
train_dqn.py

Finalized training script entry point for the DQN agent. Orchestrates 
network initialization, target network sync, training loop, and 
checkpoint saving.

Note: Full training loop (replay buffer, epsilon-greedy, optimizer 
step) is implemented by Member 2 in dqn_agent.py. This script serves 
as the environment/simulation-side entry point and checkpoint handler.
"""

import os
import torch
from dqn_network import DQNNetwork, sync_target_network
from pricing_env import PricingEnv


def save_checkpoint(model, path="models/dqn_best.pt"):
    """
    Saves the trained model's weights to disk. 
    NOTE: models/ is excluded from git via .gitignore.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, path="models/dqn_best.pt"):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


if __name__ == "__main__":
    env = PricingEnv()
    policy_net = DQNNetwork(state_dim=2, action_dim=env.action_space.n)
    target_net = DQNNetwork(state_dim=2, action_dim=env.action_space.n)
    sync_target_network(policy_net, target_net)

    print("DQN network and target network initialized.")
    print("Full training loop handled in dqn_agent.py (Member 2).")

    # Placeholder checkpoint save to verify save/load pipeline works
    save_checkpoint(policy_net)