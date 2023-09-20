import numpy as np
from typing import Union

"""
Single class and multiclass environment functions

Noises functions are to be wrapped in a lambda
"""

n_features = 2
n_classes = 3

class_map = {
    (0, 0): 0,
    (0, 1): 0,
    (1, 0): 1,
    (1, 1): 2
}

user_prob_map = {
    (0, 0): 0.3,
    (0, 1): 0.7,
    (1, 0): 1,
    (1, 1): 1
}


def _gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2 / c**2)


def get_bids():
    return np.linspace(1, 10, num=100)


def get_prices():
    return np.linspace(100, 700, num=5)


def daily_clicks_curve(bid: Union[float, np.ndarray]):
    """
    Expected number of clicks given a bid
    """
    return np.floor(-20 * (bid ** 2) + 251 * bid - 181)


def daily_clicks_curve_abrupt(bid: Union[float, np.ndarray], day: int):
    """
    Expected number of clicks given a bid
    """
    if day <= 120:
        return np.floor(-20 * (bid ** 2) + 251 * bid - 181)
    elif day <= 280:
        return np.floor(-20 * ((bid - 1.5) ** 2) + 251 * (bid - 1.5) - 181)
    else:
        return np.floor(-20 * ((bid - 3) ** 2) + 251 * (bid - 3) - 181)


def click_curve_noise(rng: np.random.Generator, size):
    """
    Simple noise generator for the daily clicks curve
    """
    return rng.normal(0, 40, size=size)


def click_cumulative_cost(bid: Union[float, np.ndarray]):
    """
    Cumulative cost of clicks given bid
    """
    # return -bid * (bid - 14) * 50
    # return -50*bid**2 + 700*bid
    return -68 * (bid ** 2) + 968 * bid - 850


def click_cumulative_cost_noise(rng: np.random.Generator, size):
    """
    Noise for the cumulative cost of clicks
    """
    return rng.normal(0, 40, size=size)


def click_conversion_rate(price: Union[float, np.ndarray]):
    """
    Conversion rate curve, since environment requires a dictionary, and we
    only require 5 prices per curve, we can choose 5 equally separated prices
    """

    # return (-3 * ((price - 6) ** 2) + 30) / 35
    return -5.4167e-06 * (price ** 2) + 0.00467 * price - 0.3125


def click_conversion_rate_abrupt(price: Union[float, np.ndarray], day: int):
    """
    Conversion rate curve, since environment requires a dictionary, and we
    only require 5 prices per curve, we can choose 5 equally separated prices
    """
    if day <= 120:
        return -5.4167e-06 * ((price - 100) ** 2) + 0.00467 * (price - 100) - 0.3125
    elif day <= 280:
        return -5.4167e-06 * ((price - 200) ** 2) + 0.00467 * (price - 200) - 0.3125
    else:
        return -5.4167e-06 * (price ** 2) + 0.00467 * price - 0.3125


def daily_clicks_curve_multiclass(bid: float, customer_class: int):
    match customer_class:
        # casual customer
        case 0:
            return np.floor((-(bid - 2) ** 2 + bid) * 6)
        case 1:
            return np.floor(-(bid - 2) * (bid - 6) * 8)
        case 2:
            return np.floor(-(bid - 4) * (bid - 8) * 6)
    return 0


def click_conversion_rate_multiclass(price: float, customer_class: int):
    """
    Conversion rate curve, since environment requires a dictionary, and we
    only require 5 prices per curve, we can choose 5 equally separated prices
    in our curve for multiclass
    """
    match customer_class:
        case 0:
            return -5.4167e-06 * (price ** 2) + 0.00467 * price - 0.3125
        case 1:
            return -3.5e-6 * (price ** 2) + 0.001424 * price + 0.7576
        case 2:
            return _gaussian(price, 0.8, 600, 400)
    return 0


def click_cumulative_cost_multiclass(bid: float, customer_class: int):
    match customer_class:
        case 0:
            return -bid * (bid - 10) * 30
        case 1:
            return -bid * (bid - 12) * 100
        case 2:
            return -bid * (bid - 11) * 100
    return 0
