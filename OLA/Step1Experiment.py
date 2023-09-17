import numpy as np

from OLA.environments import SingleClassEnvironment
from OLA.base_learners import Step1UCBLearner
from OLA.base_learners import Step1TSLearner
from OLA.environment_properties import click_curve_noise
from OLA.environment_properties import daily_clicks_curve
from OLA.environment_properties import click_cumulative_cost_noise
from OLA.environment_properties import click_cumulative_cost
from OLA.simulators import simulate_single_class
from OLA.simulators import plot_single_class_sim_result


def env_init_step1():
    N = np.vectorize(daily_clicks_curve)
    en = click_curve_noise
    C = np.vectorize(click_cumulative_cost)
    ec = click_cumulative_cost_noise
    rng = np.random
    A = {
        4: 0.51,
        5: 0.77,
        6: 0.85,
        7: 0.77,
        8: 0.51,
    }

    return SingleClassEnvironment(N, en, C, ec, A, rng)


def ucb1_learner_init(env: SingleClassEnvironment):
    bids = np.array([4, 5, 6, 7, 8])
    price = np.array([4, 5, 6, 7, 8])
    return Step1UCBLearner(env, bids, price)


if __name__ == '__main__':
    sim_object = simulate_single_class(env_init_step1, ucb1_learner_init, 365)
    plot_single_class_sim_result(sim_object)
