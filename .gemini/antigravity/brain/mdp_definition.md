# MDP Definition

- State: [remaining_inventory, days_until_departure]
- Action: Discrete price level bins
- Transition: Demand model determines units sold based on price and time
- Reward: price_level * units_sold
- Horizon: Episode ends when inventory = 0 or days_until_departure = 0