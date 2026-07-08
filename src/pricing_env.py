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