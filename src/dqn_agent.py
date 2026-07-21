import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy


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
    DQN agent with epsilon-greedy exploration, a target network for
    stable Q-learning targets, and a full training loop using
    experience replay.
    """

    def __init__(self, state_dim=2, n_actions=10, hidden_size=64, lr=1e-3,
                 gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                 target_update_freq=200):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.train_step_count = 0

        self.policy_net = DQNNetwork(state_dim, n_actions, hidden_size)
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def act(self, obs):
        """Epsilon-greedy action selection for training."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return self.act_greedy(obs)

    def act_greedy(self, obs):
        """Select the action with highest predicted Q-value (no exploration)."""
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        """Hard update: copy policy_net weights into target_net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train_step(self, buffer, batch_size=64):
        """Sample a batch from the replay buffer and perform one gradient step."""
        if len(buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32)

        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(1)[0]
            td_target = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        loss = self.loss_fn(q_values, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from pricing_env import PricingEnv
    from replay_buffer import ReplayBuffer

    env = PricingEnv()
    agent = DQNAgent()
    buffer = ReplayBuffer(capacity=5000)

    # Fill buffer with a few random episodes
    obs, info = env.reset()
    for _ in range(300):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs if not done else env.reset()[0]

    # Run a few training steps
    for i in range(10):
        loss = agent.train_step(buffer, batch_size=32)
        if loss is not None:
            print(f"Step {i}: loss={loss:.4f}, epsilon={agent.epsilon:.3f}")

    agent.decay_epsilon()
    print("Final epsilon after decay:", agent.epsilon)