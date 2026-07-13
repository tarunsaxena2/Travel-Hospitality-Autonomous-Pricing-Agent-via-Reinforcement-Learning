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
        remaining_inventory, days_until_departure = observation

        # Reduce price level by 10% relative to max, each day
        discount_factor = 0.9 ** (self.start_price_level - self.current_price_level)
        new_price_level = int(self.start_price_level * discount_factor)
        self.current_price_level = max(0, min(new_price_level, self.num_price_levels - 1))

        return self.current_price_level

    def reset(self):
        """Reset the agent's internal price tracking for a new episode."""
        self.current_price_level = self.start_price_level
    
    def get_price_value(self):
        """Returns the normalized price level (0 to 1) this agent uses."""
        return self.price_level_index / (self.num_price_levels - 1)