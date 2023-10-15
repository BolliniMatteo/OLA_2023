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


def daily_clicks_curve_multiclass(bid: Union[float, np.ndarray], customer_class: int):
    if customer_class == 0:
        # casual customer
        return _gaussian(bid, 800, 3, 4)
    elif customer_class == 1:
        return _gaussian(bid, 700, 6, 8)
    else:
        return _gaussian(bid, 720, 8, 16)


def click_conversion_rate_multiclass(price: Union[float, np.ndarray], customer_class: int):
    """
    Conversion rate curve, since environment requires a dictionary, and we
    only require 5 prices per curve, we can choose 5 equally separated prices
    in our curve for multiclass
    """
    if customer_class == 0:
        return -5.4167e-06 * (price ** 2) + 0.00467 * price - 0.3125
    elif customer_class == 1:
        return -3.5e-6 * (price ** 2) + 0.001424 * price + 0.7576
    else:
        return _gaussian(price, 0.8, 600, 400)


def click_cumulative_cost_multiclass(bid: Union[float, np.ndarray], customer_class: int):
    if customer_class == 0:
        return _gaussian(bid, 1720, 4, 10)
    elif customer_class == 1:
        return _gaussian(bid, 2300, 5, 16)
    else:
        return _gaussian(bid, 2120, 6, 12)
