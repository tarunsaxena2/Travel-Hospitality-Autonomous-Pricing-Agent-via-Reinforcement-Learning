def clip_price_level(price_level, min_level=0, max_level=9):
    """Ensure the selected price level stays within valid bounds."""
    return max(min_level, min(price_level, max_level))


def is_valid_action(price_level, inventory, days_remaining):
    """
    Returns False if the action is degenerate — e.g. selling when
    there is no inventory left or no days remaining to sell.
    """
    if inventory <= 0:
        return False
    if days_remaining <= 0:
        return False
    return True


if __name__ == "__main__":
    print("clip_price_level(12) ->", clip_price_level(12))
    print("clip_price_level(-3) ->", clip_price_level(-3))
    print("is_valid_action(5, inventory=0, days_remaining=3) ->",
          is_valid_action(5, inventory=0, days_remaining=3))
    print("is_valid_action(5, inventory=10, days_remaining=0) ->",
          is_valid_action(5, inventory=10, days_remaining=0))