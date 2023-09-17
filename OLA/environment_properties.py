import numpy as np

"""
Single class and multiclass environment functions
"""


def daily_clicks_curve(bid: float):
    """
    Expected number of clicks given a bid, the bid may be between 4 and 8
    """
    return np.floor(-3 * (bid - 6) ** 2 + 30)


def daily_clicks_curve_multiclass(bid: float, customer_class: int):
    match customer_class:
        # casual customer
        case 1:
            return np.floor((-(bid - 2) ** 2 + bid) * 6)
        case 2:
            return np.floor(-(bid - 2) * (bid - 6) * 8)
        case 3:
            return np.floor(-(bid - 4) * (bid - 8) * 6)
    return 0


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


def click_conversion_rate_multiclass(price: float, customer_class: int):
    """
    Conversion rate curve, since environment requires a dictionary, and we
    only require 5 prices per curve, we can choose 5 equally separated prices
    in our curve for multiclass
    """
    match customer_class:
        case 1:
            return (-(price - 2) ** 2 + price) / 3
        case 2:
            return (-(price - 2) * (price - 6)) / 4
        case 3:
            return (-(price - 4) * (price - 8)) / 5
    return 0


def click_cumulative_cost(bid: float):
    """
    Cumulative cost of clicks given bid
    """
    return -bid * (bid - 14) * 50


def click_cumulative_cost_multiclass(bid: float, customer_class: int):
    match customer_class:
        case 1:
            return -bid * (bid - 10) * 30
        case 2:
            return -bid * (bid - 12) * 100
        case 3:
            return -bid * (bid - 11) * 100
    return 0


def click_cumulative_cost_noise():
    """
    Noise for the cumulative cost of clicks
    """
    return np.random.normal(10, 4)
