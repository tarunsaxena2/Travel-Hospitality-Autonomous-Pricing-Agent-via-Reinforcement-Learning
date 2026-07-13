import numpy as np

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

    def act(self, observation):
        remaining_inventory, days_until_departure = observation

        discount_factor = 0.9 ** (self.start_price_level - self.current_price_level)
        new_price_level = int(self.start_price_level * discount_factor)
        self.current_price_level = max(0, min(new_price_level, self.num_price_levels - 1))

        return self.current_price_level

    def reset(self):
        """Reset the agent's internal price tracking for a new episode."""
        self.current_price_level = self.start_price_level