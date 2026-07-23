"""
dqn_network.py

Defines the neural network architecture used by the DQN agent to 
approximate Q-values for the dynamic pricing environment.
"""

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """
    Simple feedforward neural network for approximating Q-values.

    Input: state features (remaining_inventory, days_until_departure)
    Output: Q-value for each discrete price action
    """

    def __init__(self, state_dim=2, action_dim=10, hidden_dim=64):
        super(DQNNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)



def sync_target_network(policy_net, target_net):
    """
    Performs a hard update: copies all weights from the policy 
    network to the target network.
    
    Called periodically (every N training steps) to stabilize 
    DQN training by keeping the target Q-values fixed for a while.
    """
    target_net.load_state_dict(policy_net.state_dict())

