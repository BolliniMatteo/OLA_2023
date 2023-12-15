# I optimize the burn in for the UCB with known classes

import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process.kernels
import warnings
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import new_environment_properties as ep
from OLA.multi_class_learners import Step4UCBRealClassesLearner
from OLA.environments import MultiClassEnvironment
from OLA.simulators import simulate_multi_class

warnings.filterwarnings("ignore")


def env_init_step4(rng: np.random.Generator):
    n_features = ep.n_features
    class_map = ep.class_map
    user_prob_map = ep.user_prob_map
    n = ep.daily_clicks_curve_multiclass
    en = lambda: ep.daily_click_curve_noise(rng, 1)[0]
    c = ep.click_cumulative_cost_multiclass
    ec = lambda: ep.advertising_costs_curve_noise(rng, 1)[0]
    a = ep.click_conversion_rate_multiclass
    return MultiClassEnvironment(n_features, class_map, user_prob_map, n, en, c, ec, a, ep.get_production_cost(), rng)


def gpucb_known_learner_init(env: MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                             kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float,
                             burn_in: int, ucb_constant: float):
    return Step4UCBRealClassesLearner(env, bids, prices, kernel, alpha, beta, burn_in, ucb_constant)


def simulate_burn_in(burn_in: int, kernel: sklearn.gaussian_process.kernels.Kernel,
                     alpha: float, beta: float, ucb_constant: float, rng: np.random.Generator, n_runs: int, T: int):
    learner_init_gpucb_known = lambda env: gpucb_known_learner_init(env, ep.get_bids(), ep.get_prices(),
                                                                    kernel, alpha, beta,
                                                                    burn_in, ucb_constant)
    res_gpucb_known = simulate_multi_class(lambda: env_init_step4(rng),
                                           learner_init_gpucb_known, T, n_runs).aggregate_results
    return res_gpucb_known.cum_rewards_mean[-1]


def main():
    seed = 5000
    rng = np.random.default_rng(seed=seed)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
    alpha = 10
    beta = 110
    T = 365
    n_runs = 5
    c = 0.2

    burn_in_values = np.array([i*14 + 20 for i in range(6)])
    results = np.array([simulate_burn_in(burn_in, kernel, alpha, beta, c, rng, n_runs, T)
                        for burn_in in burn_in_values])
    plt.plot(burn_in_values, results)
    plt.scatter(burn_in_values, results)
    best_index = np.argmax(results)
    plt.scatter(burn_in_values[best_index], results[best_index], color='r')
    plt.show()
    print("Best burn in:", burn_in_values[best_index])


if __name__ == '__main__':
    main()
