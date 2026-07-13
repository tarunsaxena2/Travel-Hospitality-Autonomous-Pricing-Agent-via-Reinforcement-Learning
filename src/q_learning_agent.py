import numpy as np


class QLearningAgent:
    """Tabular Q-Learning agent for the pricing environment."""

    def __init__(self, inventory_bins=10, days_bins=10, n_actions=10,
                 max_inventory=100, max_days=30,
                 learning_rate=0.1, discount_factor=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.inventory_bins = inventory_bins
        self.days_bins = days_bins
        self.n_actions = n_actions
        self.max_inventory = max_inventory
        self.max_days = max_days

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = np.zeros((inventory_bins, days_bins, n_actions))

    def discretize_state(self, inventory, days_remaining):
        inv_bucket = int(np.clip(
            (inventory / self.max_inventory) * self.inventory_bins,
            0, self.inventory_bins - 1
        ))
        day_bucket = int(np.clip(
            (days_remaining / self.max_days) * self.days_bins,
            0, self.days_bins - 1
        ))
        return inv_bucket, day_bucket

    def act(self, obs):
        """Epsilon-greedy action selection."""
        inv_bucket, day_bucket = self.discretize_state(obs[0], obs[1])

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[inv_bucket, day_bucket]))

    def update(self, obs, action, reward, next_obs, done):
        """Bellman update: Q(s,a) += lr * (reward + gamma * max(Q(s')) - Q(s,a))"""
        inv, day = self.discretize_state(obs[0], obs[1])
        next_inv, next_day = self.discretize_state(next_obs[0], next_obs[1])

        current_q = self.q_table[inv, day, action]
        max_next_q = 0 if done else np.max(self.q_table[next_inv, next_day])

        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q

        self.q_table[inv, day, action] = current_q + self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from pricing_env import PricingEnv

    env = PricingEnv()
    agent = QLearningAgent()

    obs, info = env.reset()
    action = agent.act(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    agent.update(obs, action, reward, next_obs, terminated or truncated)

    print("Q-table shape:", agent.q_table.shape)
    print("Sample updated Q-value:", agent.q_table[agent.discretize_state(obs[0], obs[1])][action])
    print("Epsilon after one decay step:", agent.epsilon)