import numpy as np

from OLA.environments import SingleClassEnvironment
from OLA.base_learners import Step1UCBLearner
from OLA.base_learners import Step1TSLearner
import new_environment_properties as ep
from OLA.simulators import simulate_single_class
from OLA.simulators import plot_single_class_sim_result


def env_init_step1(rng: np.random.Generator):
    N = ep.daily_clicks_curve
    en = lambda: ep.daily_click_curve_noise(rng, None) # size None returns a scalar, size 1 an array of a single element
    C = ep.click_cumulative_cost
    ec = lambda: ep.advertising_costs_curve_noise(rng, None)
    A = ep.click_conversion_rate

    return SingleClassEnvironment(N, en, C, ec, A, ep.get_production_cost(), rng)


def ucb1_learner_init(env: SingleClassEnvironment, bids: np.ndarray, prices: np.ndarray):
    return Step1UCBLearner(env, bids, prices)


if __name__ == '__main__':
    seed = 1000
    bids = ep.get_bids()
    prices = ep.get_prices()
    rng = np.random.default_rng(seed=seed)
    T = 365
    sim_object_ucb1 = simulate_single_class(lambda: env_init_step1(rng),
                                            bids, prices,
                                            lambda env, bids, prices: Step1UCBLearner(env, bids, prices),
                                            T)
    sim_object_ts = simulate_single_class(lambda: env_init_step1(rng),
                                          bids, prices,
                                          lambda env, bids, prices: Step1TSLearner(env, bids, prices, rng), T)
    plot_single_class_sim_result(sim_object_ucb1)
    plot_single_class_sim_result(sim_object_ts)
