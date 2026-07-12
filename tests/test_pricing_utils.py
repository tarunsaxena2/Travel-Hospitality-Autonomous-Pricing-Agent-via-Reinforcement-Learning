import sys
sys.path.append('../src')
from pricing_utils import clip_price_level, is_valid_action


def test_clip_price_level_within_bounds():
    assert clip_price_level(5) == 5


def test_clip_price_level_above_max():
    assert clip_price_level(15) == 9


def test_clip_price_level_below_min():
    assert clip_price_level(-4) == 0


def test_invalid_action_zero_inventory():
    assert is_valid_action(price_level=5, inventory=0, days_remaining=10) is False


def test_invalid_action_zero_days():
    assert is_valid_action(price_level=5, inventory=10, days_remaining=0) is False


def test_valid_action():
    assert is_valid_action(price_level=5, inventory=10, days_remaining=5) is True


if __name__ == "__main__":
    test_clip_price_level_within_bounds()
    test_clip_price_level_above_max()
    test_clip_price_level_below_min()
    test_invalid_action_zero_inventory()
    test_invalid_action_zero_days()
    test_valid_action()
    print("All edge case tests passed!")