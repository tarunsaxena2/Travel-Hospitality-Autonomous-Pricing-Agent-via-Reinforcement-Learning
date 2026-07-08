import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PricingEnv(gym.Env):
    """
    Custom Gymnasium environment for dynamic pricing in travel/hospitality.
    
    State: [remaining_inventory, days_until_departure]
    Action: discrete price level bins
    """

    def __init__(self, max_inventory=100, max_days=30, num_price_levels=10):
        super().__init__()

        self.max_inventory = max_inventory
        self.max_days = max_days
        self.num_price_levels = num_price_levels

        # Observation space: [remaining_inventory, days_until_departure]
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([max_inventory, max_days]),
            dtype=np.float32
        )

        # Action space: discrete price level bins
        self.action_space = spaces.Discrete(num_price_levels)

        # State variables (set properly in reset())
        self.remaining_inventory = None
        self.days_until_departure = None
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize state at the start of a new episode
        self.remaining_inventory = self.max_inventory
        self.days_until_departure = self.max_days

        observation = np.array(
            [self.remaining_inventory, self.days_until_departure],
            dtype=np.float32
        )
        info = {}

        return observation, info
    def step(self, action):
        # Convert discrete action to a price level (placeholder logic for now)
        price_level = action / (self.num_price_levels - 1)  # normalized 0 to 1

        # Placeholder demand logic (Member 1 will refine this on Day 4)
        units_sold = max(0, int(np.random.poisson(lam=5 * (1 - price_level))))
        units_sold = min(units_sold, self.remaining_inventory)

        # Update state
        self.remaining_inventory -= units_sold
        self.days_until_departure -= 1

        # Reward = revenue generated this step
        reward = price_level * units_sold

        # Episode ends if no inventory left or no days left
        terminated = self.remaining_inventory <= 0 or self.days_until_departure <= 0
        truncated = False

        observation = np.array(
            [self.remaining_inventory, self.days_until_departure],
            dtype=np.float32
        )
        info = {"units_sold": units_sold}

        return observation, reward, terminated, truncated, info