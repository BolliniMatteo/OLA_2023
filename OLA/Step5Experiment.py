import numpy as np

from OLA.environments import SingleClassEnvironmentNonStationary
from OLA.base_learners import Step5UCBLearner
from OLA.base_learners import Step5UCBWINLearner
import new_environment_properties as ep
from OLA.simulators import simulate_single_class
from OLA.simulators import plot_single_class_sim_result


def env_init_step5(rng: np.random.Generator):
    N = ep.daily_clicks_curve
    en = lambda: ep.daily_click_curve_noise(rng, None) # size None returns a scalar, size 1 an array of a single element
    C = ep.click_cumulative_cost
    ec = lambda: ep.advertising_costs_curve_noise(rng, None)
    A = ep.conversion_rate_three_phases

    return SingleClassEnvironmentNonStationary(N, en, C, ec, A, ep.get_production_cost(), rng)


if __name__ == '__main__':
    seed = 1000
    bids = ep.get_bids()
    prices = ep.get_prices()
    rng = np.random.default_rng(seed=seed)
    T = 365
    """
    print("-----Simple UCB-----")
    sim_object_ucb1 = simulate_single_class(lambda: env_init_step5(rng),
                                            bids, prices,
                                            lambda env, bids, prices: Step5UCBLearner(env, bids, prices),
                                            T, n_runs=300)
    plot_single_class_sim_result(sim_object_ucb1)
    """
    print("-----Window UCB-----")
    win_size = 40
    sim_object_win = simulate_single_class(lambda: env_init_step5(rng),
                                            bids, prices,
                                            lambda env, bids, prices: Step5UCBWINLearner(env, bids, prices, win_size),
                                            T, n_runs=300)
    plot_single_class_sim_result(sim_object_win)
