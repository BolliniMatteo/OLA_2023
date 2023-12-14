import numpy as np
from typing import Union

"""
Single class and multiclass environment functions

Noises functions are to be wrapped in a lambda
"""

n_features = 2
n_classes = 3

class_map = {
    (0, 0): 1,
    (0, 1): 2,
    (1, 0): 0,
    (1, 1): 1
}

user_prob_map = {
    (0, 0): 0.3,
    (0, 1): 1,
    (1, 0): 1,
    (1, 1): 0.7
}


def get_bids():
    return np.linspace(1, 10, num=100)


def get_prices():
    return np.linspace(30, 50, num=5)


def get_production_cost():
    return 15


def daily_clicks_curve_multiclass(bid: Union[float, np.ndarray], customer_class: int):
    if customer_class == 0:
        # return np.floor(70 * (np.tanh(0.6 * (bid - 5)) + 1))
        return 70 * (np.tanh(0.7 * (bid - 5)) + 1)
    elif customer_class == 1:
        # return np.floor(70 / (1 + np.exp(-1.2 * (bid - 5))))
        return 65 / (1 + np.exp(-1.3 * (bid - 4.5)))
    else:
        # return np.floor(60 * (np.tanh(0.4 * (bid - 5)) + 1))
        return 30 * (np.tanh(0.4 * (bid - 5)) + 1) + 40 / (1 + np.exp(-1.2 * (bid - 5)))


def click_cumulative_cost_multiclass(bid: Union[float, np.ndarray], customer_class: int):
    if customer_class == 0:
        return 300 * (np.tanh(0.6 * (bid - 6.5)) + 1)
    elif customer_class == 1:
        return 250 / (1 + np.exp(-1.3 * (bid - 6))) + 100 * (np.tanh(0.7 * (bid - 6)) + 1)
        # return 300 / (1 + np.exp(-1.3 * (bid - 6)))
    else:
        return 250 * (np.tanh(0.4 * (bid - 6)) + 1)


def click_conversion_rate_multiclass(price: Union[float, np.ndarray], customer_class: int):
    if customer_class == 0:
        # w1, w2, w3 = (-0.00135000, 0.0775, -0.2700)
        w1, w2, w3 = (0.00155000, -0.15450000, 4.17000000)
        return w1 * (price ** 2) + w2 * price + w3
    elif customer_class == 1:
        w1, w2, w3, w4 = (-0.00015667, 0.01905000, -0.78683333, 11.52000000)
        return w1 * (price ** 3) + w2 * (price ** 2) + w3 * price + w4
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


def conversion_rate_three_phases(price: Union[float, np.ndarray], t: int):
    if t <= 30*8 + 4:
        return click_conversion_rate_multiclass(price, single_cass)
    # there is not "concave" constraint for the phases
    if t <= 30*10 + 20:
        # beginning of the academic year: very large demand
        w1, w2, w3 = (0.0006, -0.0620, 2.2700)
        return w1 * (price ** 2) + w2 * price + w3
    # Christmas period: larger demand... but for not too large prices
    # w1, w2, w3 = (0.00155000, -0.1535, 4.0300)
    # w1, w2, w3 = (0.00065000, -0.07850000, 2.72000000)
    w1, w2, w3 = (0.00015000, -0.04350000, 2.12000000)
    return w1 * (price ** 2) + w2 * price + w3


def conversion_rate_high_frequency_phases(price: Union[float, np.ndarray], t: int):
    # length of each phase
    # lengths = np.array([5, 6, 7, 8, 5])
    lengths = np.array([10, 10, 10, 10, 10])
    ends = np.cumsum(lengths)
    period = np.sum(lengths)
    season_t = t % period
    phase = np.argwhere(season_t < ends)[0, 0]
    # print(f"[DEBUG] {t=}, {phase=}")
    return conversion_rate_five_phases(price, phase)


def conversion_rate_five_phases(price: Union[float, np.ndarray], phase: int):
    """
    This is NOT the final function for step 6!
    Use conversion_rate_high_frequency_phases instead
    :param price:
    :param phase:
    :return:
    """
    # obs: we have 5 prices and each phase must have a different optimal price (optimize alpha*(price-cost))
    if phase == 0:
        # w1, w2, w3, w4, w5 = (-0.00000667, 0.0010, -0.0578, 1.5450, -15.3000)
        w1, w2, w3, w4, w5 = (-0.00000667, 0.0010, -0.0578, 1.5450, -15.7000)
        return w1 * (price ** 4) + w2 * (price ** 3) + w3 * (price ** 2) + w4 * price + w5
    if phase == 1:
        # w1, w2, w3, w4 = (-0.00015667, 0.01905000, -0.78683333, 11.52000000)
        # return w1 * (price ** 3) + w2 * (price ** 2) + w3 * price + w4
        w1, w2, w3 = (0.00155000, -0.15450000, 4.17000000)
        return w1 * (price ** 2) + w2 * price + w3
    if phase == 2:
        # w1, w2, w3 = (0.00155000, -0.1535, 4.0300)
        # return w1 * (price ** 2) + w2 * price + w3
        return conversion_rate_three_phases(price, 350)
    if phase == 3:
        w1, w2, w3 = (0.0006, -0.0620, 2.0700)
        return w1 * (price ** 2) + w2 * price + w3
    w1, w2, w3 = (-0.00075000, 0.0345, 0.2700)
    return w1 * (price ** 2) + w2 * price + w3
