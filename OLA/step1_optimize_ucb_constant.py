import matplotlib.pyplot as plt
import numpy as np
from typing import Union

from OLA.environments import SingleClassEnvironment,SingleClassEnvironmentHistory
from OLA.base_learners import Step1UCBLearner
from OLA.base_learners import Step1TSLearner
import new_environment_properties as ep
from OLA.simulators import simulate_single_class, plot_multiple_single_class_results
from OLA.simulators import plot_single_class_sim_result


def env_init_step1(rng: np.random.Generator):
    N = ep.daily_clicks_curve
    en = lambda: ep.daily_click_curve_noise(rng, None) # size None returns a scalar, size 1 an array of a single element
    C = ep.click_cumulative_cost
    ec = lambda: ep.advertising_costs_curve_noise(rng, None)
    A = ep.click_conversion_rate

    return SingleClassEnvironment(N, en, C, ec, A, ep.get_production_cost(), rng)


def simulate_constant(ucb_constant: float, n_runs: int, T: int, rng: np.random.Generator):
    sim_object_ucb1 = simulate_single_class(lambda: env_init_step1(rng),
                                            ep.get_bids(), ep.get_prices(),
                                            lambda env, bids, prices: Step1UCBLearner(env, bids, prices, ucb_constant),
                                            T, n_runs, None)
    return sim_object_ucb1.cum_rewards_mean[-1]


def main():
    seed = 2000
    rng = np.random.default_rng(seed=seed)
    T = 365
    n_runs = 100
    c_range = np.linspace(0.2, 3, 10)
    results = np.array([simulate_constant(c, n_runs, T, rng) for c in c_range])
    plt.plot(c_range, results)
    plt.scatter(c_range, results)
    best_index = np.argmax(results)
    plt.scatter(c_range[best_index], results[best_index], color='r')
    plt.show()


if __name__ == '__main__':
    main()
