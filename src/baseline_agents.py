import numpy as np


class FixedPriceAgent:
    """
    Baseline agent that always selects the same fixed price level 
    for the entire season, regardless of state.
    
    This serves as the simplest possible baseline — a hotel/airline 
    that never changes its price throughout the booking season.
    """

    def __init__(self, price_level_index=5, num_price_levels=10):
        """
        price_level_index: which discrete action (price bin) to always pick.
        Default is roughly the mid-range price.
        """
        self.price_level_index = price_level_index
        self.num_price_levels = num_price_levels

    def act(self, observation):
        """
        Returns the same fixed action regardless of the observation.
        """
        return self.price_level_index
    
    def get_price_value(self):
        """Returns the normalized price level (0 to 1) this agent uses."""
        return self.price_level_index / (self.num_price_levels - 1)