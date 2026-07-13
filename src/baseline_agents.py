import numpy as np


class FixedPriceAgent:
    """
    Baseline agent that always selects the same fixed price level 
    for the entire season, regardless of state.
    
    This serves as the simplest possible baseline — a hotel/airline 
    that never changes its price throughout the booking season.
    """

    def __init__(self, price_level_index=5, num_price_levels=10):
        self.price_level_index = price_level_index
        self.num_price_levels = num_price_levels

    def act(self, observation):
        return self.price_level_index

    def get_price_value(self):
        """Returns the normalized price level (0 to 1) this agent uses."""
        return self.price_level_index / (self.num_price_levels - 1)


class TimeBasedDiscountAgent:
    """
    Baseline agent that reduces price by 10% each day as the 
    deadline (days_until_departure) approaches.
    
    Simulates common airline/hotel behavior: prices drop as the 
    departure date nears to encourage last-minute bookings and 
    clear remaining inventory.
    """

    def __init__(self, start_price_level=9, num_price_levels=10):
        self.start_price_level = start_price_level
        self.num_price_levels = num_price_levels
        self.current_price_level = start_price_level
        self.days_elapsed = 0

    def act(self, observation):
        remaining_inventory, days_until_departure = observation

        discount_factor = 0.9 ** self.days_elapsed
        new_price_level = int(round(self.start_price_level * discount_factor))
        self.current_price_level = max(0, min(new_price_level, self.num_price_levels - 1))

        self.days_elapsed += 1

        return self.current_price_level

    def reset(self):
        """Reset the agent's internal price tracking for a new episode."""
        self.current_price_level = self.start_price_level
        self.days_elapsed = 0