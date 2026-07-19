import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DQNNetwork(nn.Module):
    """
    Feedforward network approximating Q-values for each discrete
    price action, given the continuous state (inventory, days_remaining).
    """

    def __init__(self, state_dim=2, n_actions=10, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """
    DQN agent wrapping the network, loss function, and optimizer.
    Replay buffer, target network, and training loop added in later commits.
    """

    def __init__(self, state_dim=2, n_actions=10, hidden_size=64, lr=1e-3):
        self.n_actions = n_actions
        self.policy_net = DQNNetwork(state_dim, n_actions, hidden_size)

        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def act_greedy(self, obs):
        """Select the action with highest predicted Q-value (no exploration)."""
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())


if __name__ == "__main__":
    agent = DQNAgent()
    sample_state = np.array([50.0, 15.0], dtype=np.float32)

    q_vals = agent.policy_net(torch.tensor(sample_state).unsqueeze(0))
    print("Q-values for sample state:", q_vals)

    action = agent.act_greedy(sample_state)
    print("Greedy action:", action)

    print("Loss function:", agent.loss_fn)
    print("Optimizer:", agent.optimizer)