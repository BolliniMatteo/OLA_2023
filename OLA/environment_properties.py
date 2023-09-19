import numpy as np
from typing import Union

"""
Single class and multiclass environment functions

Noises functions are to be wrapped in a lambda
"""


def get_bids():
    return np.linspace(1, 10, num=100)


def get_prices():
    return np.linspace(100, 700, num=5)


def daily_clicks_curve(bid: Union[float, np.ndarray]):
    """
    Expected number of clicks given a bid
    """
    return np.floor(-20 * (bid ** 2) + 251 * bid - 181)


def daily_clicks_curve_abrupt(bid: Union[float, np.ndarray], phase: int):
    """
    Expected number of clicks given a bid
    """
    match phase:
        case 1:
            return np.floor(-20 * (bid ** 2) + 251 * bid - 181)
        case 2:
            return np.floor(-20 * ((bid - 1.5) ** 2) + 251 * (bid - 1.5) - 181)
        case 3:
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


def daily_clicks_curve_multiclass(bid: float, customer_class: int):
    """
    match customer_class:
        # casual customer
        case 1:
            return np.floor((-(bid - 2) ** 2 + bid) * 6)
        case 2:
            return np.floor(-(bid - 2) * (bid - 6) * 8)
        case 3:
            return np.floor(-(bid - 4) * (bid - 8) * 6)
    return 0
    """


def click_conversion_rate_multiclass(price: float, customer_class: int):
    """
    Conversion rate curve, since environment requires a dictionary, and we
    only require 5 prices per curve, we can choose 5 equally separated prices
    in our curve for multiclass
    """
    """
    match customer_class:
        case 1:
            return (-(price - 2) ** 2 + price) / 3
        case 2:
            return (-(price - 2) * (price - 6)) / 4
        case 3:
            return (-(price - 4) * (price - 8)) / 5
    return 0
    """


def click_cumulative_cost_multiclass(bid: float, customer_class: int):
    """
    match customer_class:
        case 1:
            return -bid * (bid - 10) * 30
        case 2:
            return -bid * (bid - 12) * 100
        case 3:
            return -bid * (bid - 11) * 100
    return 0
    """
