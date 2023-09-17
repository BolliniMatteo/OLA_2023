import numpy as np

"""
Single class environment functions
"""


def daily_clicks_curve(bid: float):
    """
    Expected number of clicks given a bid, the bid may be between 4 and 8
    """
    return -3 * (bid - 6) ** 2 + 30


def daily_clicks_curve_multiclass(bid: float, user_class: int):
    """

    """
    match user_class:
        case 1:
            return asdasdd;
        case 2:
            return asasdasd;
        case 3:
            return asdasd;
    return


def click_curve_noise():
    """
    Simple noise generator for the daily clicks curve
    """
    return np.random.normal(2, 4)


def click_conversion_rate(price: float):
    """
    Conversion rate curve, since environment requires a dictionary, and we
    only require 5 prices per curve, we can choose 5 equally separated prices
    in our curve:
    A = {
        400: 0.51,
        500: 0.77,
        600: 0.85,
        700: 0.77,
        800: 0.51
    }
    """

    return (-3 * ((price - 6) ** 2) + 30) / 35


def click_cumulative_cost(bid: float):
    """
    Cumulative cost of clicks given bid
    """
    return -bid * (bid - 14) * 50


def click_cumulative_cost_noise():
    """
    Noise for the cumulative cost of clicks
    """
    return np.random.normal(10, 4)
