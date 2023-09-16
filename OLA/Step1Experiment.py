import numpy as np

from OLA.environments import SingleClassEnvironment

"""
True expected number of clicks curve given bid
"""


def daily_clicks_curve_step1(bid: float):
    return -3 * (bid - 6) ** 2 + 30


def click_curve_noise_step1():
    return np.random.normal(2, 4)


def click_conversion_rate(bid: float):
    return (-3 * ((bid - 6) ** 2) + 30) / 35


def click_cumulative_cost_step1(bid: float):
    return -bid * (bid - 14) * 50


def click_cumulative_noise_step1():
    return np.random.normal(10, 4)


if __name__ == '__main__':
    N = daily_clicks_curve_step1
    en = click_curve_noise_step1
    C = click_cumulative_cost_step1
    ec = click_curve_noise_step1
    rng = np.random
    # For completeness' sake, the conversion rate curve
    # is (-3((x - 6) ** 2) + 30) / 35
    A = {
        400: 0.51,
        500: 0.77,
        600: 0.85,
        700: 0.77,
        800: 0.51
    }

    env = SingleClassEnvironment(N, en, C, ec, A, rng)
