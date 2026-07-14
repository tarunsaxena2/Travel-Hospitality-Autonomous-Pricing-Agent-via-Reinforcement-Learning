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

class DemandBasedAgent:
    """
    Baseline agent that adjusts price based on the ratio of 
    remaining inventory to remaining time.
    
    Logic: If inventory is high relative to days left (slow sales), 
    lower the price to stimulate demand. If inventory is low relative 
    to days left (fast sales / high demand), raise the price.
    """

    def __init__(self, max_inventory=100, max_days=30, num_price_levels=10):
        self.max_inventory = max_inventory
        self.max_days = max_days
        self.num_price_levels = num_price_levels

    def act(self, observation):
        remaining_inventory, days_until_departure = observation

        # Avoid division by zero when no days are left
        days_left = max(days_until_departure, 1)

        # Ratio: how much inventory remains per day left, normalized
        inventory_time_ratio = (remaining_inventory / self.max_inventory) / (days_left / self.max_days)

        # High ratio (too much inventory for time left) -> lower price
        # Low ratio (inventory selling fast) -> higher price
        # Clamp ratio to a reasonable range before scaling to price level
        clamped_ratio = min(max(inventory_time_ratio, 0.1), 2.0)

        # Inverse relationship: higher ratio -> lower price_level
        price_level = int(round((self.num_price_levels - 1) * (1 - (clamped_ratio / 2.0))))
        price_level = max(0, min(price_level, self.num_price_levels - 1))

        return price_level