import numpy as np

from OLA.environments import SingleClassEnvironment,SingleClassEnvironmentHistory
from OLA.base_learners import Step1UCBLearner
from OLA.base_learners import Step1TSLearner
import new_environment_properties as ep
from OLA.simulators import simulate_single_class, plot_multiple_single_class_results


def env_init_step1(rng: np.random.Generator):
    N = ep.daily_clicks_curve
    en = lambda: ep.daily_click_curve_noise(rng, None)
    # size None returns a scalar, size 1 an array of a single element
    C = ep.click_cumulative_cost
    ec = lambda: ep.advertising_costs_curve_noise(rng, None)
    A = ep.click_conversion_rate

    return SingleClassEnvironment(N, en, C, ec, A, ep.get_production_cost(), rng)


if __name__ == '__main__':
    seed = 2000
    bids = ep.get_bids()
    prices = ep.get_prices()
    rng = np.random.default_rng(seed=seed)
    T = 365
    n_runs = 3000
    opt_rewards = SingleClassEnvironmentHistory(env_init_step1(rng)).clairvoyant_rewards(bids, prices, T)
    sim_object_ucb1 = simulate_single_class(lambda: env_init_step1(rng),
                                            bids, prices,
                                            lambda env, bids, prices: Step1UCBLearner(env, bids, prices, 0.4),
                                            T, n_runs)

    sim_object_ts = simulate_single_class(lambda: env_init_step1(rng),
                                          bids, prices,
                                          lambda env, bids, prices: Step1TSLearner(env, bids, prices, rng),
                                          T, n_runs)

    plot_multiple_single_class_results([sim_object_ucb1, sim_object_ts], opt_rewards, ['UCB1', 'TS'],
                                       True, '../Plots/step1.png')