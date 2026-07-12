import numpy as np


class QLearningAgent:
    """Tabular Q-Learning agent for the pricing environment."""

    def __init__(self, inventory_bins=10, days_bins=10, n_actions=10,
                 max_inventory=100, max_days=30):
        self.inventory_bins = inventory_bins
        self.days_bins = days_bins
        self.n_actions = n_actions
        self.max_inventory = max_inventory
        self.max_days = max_days

        # Q-table: shape (inventory_bucket, days_bucket, action)
        self.q_table = np.zeros((inventory_bins, days_bins, n_actions))

    def discretize_state(self, inventory, days_remaining):
        """Map continuous (inventory, days_remaining) into discrete buckets."""
        inv_bucket = int(np.clip(
            (inventory / self.max_inventory) * self.inventory_bins,
            0, self.inventory_bins - 1
        ))
        day_bucket = int(np.clip(
            (days_remaining / self.max_days) * self.days_bins,
            0, self.days_bins - 1
        ))
        return inv_bucket, day_bucket


if __name__ == "__main__":
    agent = QLearningAgent()
    print("Q-table shape:", agent.q_table.shape)
    print("Sample discretization (inv=50, days=15):",
          agent.discretize_state(50, 15))
    print("All zeros?", np.all(agent.q_table == 0))