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
    return a * np.exp(-(x - b) ** 2 / c ** 2)


def get_bids():
    return np.linspace(1, 10, num=100)


def get_prices():
    return np.linspace(30, 50, num=5)


def get_production_cost():
    return 15


def daily_clicks_curve_multiclass(bid: Union[float, np.ndarray], customer_class: int):
    if customer_class == 0:
        return np.floor(100 / (1 + np.exp(-1.2 * (bid - 5))))
    elif customer_class == 1:
        return np.floor(70 * (np.tanh(0.4 * (bid - 5)) + 1))
    else:
        return np.floor(90 * (np.tanh(0.4 * (bid - 5)) + 1))


def click_cumulative_cost_multiclass(bid: Union[float, np.ndarray], customer_class: int):
    if customer_class == 0:
        return 600 / (1 + np.exp(-1.2 * (bid - 6)))
    elif customer_class == 1:
        return 350 * (np.tanh(0.4 * (bid - 6)) + 1)
    else:
        return 400 * (np.tanh(0.4 * (bid - 6)) + 1)


def click_conversion_rate_multiclass(price: Union[float, np.ndarray], customer_class: int):
    if customer_class == 0:
        w1, w2, w3 = (-0.0005, 0.0100, 0.8500)
        return w1 * (price ** 2) + w2 * price + w3
    elif customer_class == 1:
        w1, w2, w3 = (-0.0005, 0.0050, 1.2000)
        return w1 * (price ** 2) + w2 * price + w3
    else:
        w1, w2, w3, w4, w5 = (-0.00000667, 0.0010, -0.0578, 1.5450, -15.3000)
        return w1 * (price ** 4) + w2 * (price ** 3) + w3 * (price ** 2) + w4 * price + w5


# the class used for single class environments
single_cass = 0


def daily_clicks_curve(bid: Union[float, np.ndarray]):
    return daily_clicks_curve_multiclass(bid, single_cass)


def daily_click_curve_noise(rng: np.random.Generator, size):
    return rng.normal(0, 15, size=size)


def click_cumulative_cost(bid: Union[float, np.ndarray]):
    return click_cumulative_cost_multiclass(bid, single_cass)


def advertising_costs_curve_noise(rng: np.random.Generator, size):
    return rng.normal(0, 50, size=size)


def click_conversion_rate(price: Union[float, np.ndarray]):
    return click_conversion_rate_multiclass(price, single_cass)


# TODO: update the following

def daily_clicks_curve_abrupt(bid: Union[float, np.ndarray], day: int):
    if day <= 120:
        return np.floor(-20 * (bid ** 2) + 251 * bid - 181)
    elif day <= 280:
        return np.floor(-20 * ((bid - 1.5) ** 2) + 251 * (bid - 1.5) - 181)
    else:
        return np.floor(-20 * ((bid - 3) ** 2) + 251 * (bid - 3) - 181)


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
