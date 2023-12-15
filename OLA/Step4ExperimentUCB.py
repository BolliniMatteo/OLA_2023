import numpy as np
import sklearn.gaussian_process.kernels
import warnings
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import os

import new_environment_properties as ep
from OLA.multi_class_learners import Step4UCBStatefulContextLearner, Step4UCBRealClassesLearner, Step4UCBOneClassLearner
from OLA.context_gen import StatefulContextGenerator
from OLA.environments import MultiClassEnvironment, MultiClassEnvironmentHistory
from OLA.simulators import simulate_multi_class, plot_multiple_single_class_results

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


def gpucb_unknown_learner_init(env: MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                               kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float,
                               burn_in: int, rng: np.random.Generator, bound_confidence: float, ucb_constant: float):
    context_gen = StatefulContextGenerator(env, bids, prices, kernel, alpha, rng, beta, bound_confidence)
    return Step4UCBStatefulContextLearner(env, bids, prices, kernel, alpha, beta, context_gen, burn_in, ucb_constant)


def gpucb_one_learner_init(env: MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                           kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float,
                           ucb_constant: float):
    return Step4UCBOneClassLearner(env, bids, prices, kernel, alpha, beta, ucb_constant)


def main():
    seed = 5000
    bids = ep.get_bids()
    prices = ep.get_prices()
    rng = np.random.default_rng(seed=seed)
    env_init = lambda: env_init_step4(rng)

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
    alpha = 10
    # beta should be around 110
    beta = 110
    ucb_constant = 0.2
    burn_in = 0 # should be 0
    ucb_known_classes_burn_in = 20 # optimize this with the script step_4_optimize_burn_in.py
    T = 365  # should be 365
    hoeffding_bound_confidence = 0.90

    learner_init_gpucb_known = lambda env: gpucb_known_learner_init(env, bids, prices, kernel, alpha, beta,
                                                                    ucb_known_classes_burn_in, ucb_constant)
    learner_init_gpucb_unknown = lambda env: gpucb_unknown_learner_init(env, bids, prices, kernel, alpha, beta, burn_in,
                                                                        rng, hoeffding_bound_confidence, ucb_constant)
    learner_init_gpucb_one = lambda env: gpucb_one_learner_init(env, bids, prices, kernel, alpha, beta, ucb_constant)

    n_runs = 100
    print(os.getcwd())

    print("GP-UCB learner, known classes", flush=True)
    res_gpucb_known = simulate_multi_class(env_init, learner_init_gpucb_known, T, n_runs).aggregate_results

    print("GP-UCB learner, unknown classes with context generation", flush=True)
    res_gpucb_unknown = simulate_multi_class(env_init, learner_init_gpucb_unknown, T, n_runs).aggregate_results

    print("GP-UCB learner, unknown classes using one context", flush=True)
    res_gpucb_one = simulate_multi_class(env_init, learner_init_gpucb_one, T, n_runs).aggregate_results

    opt_rewards = MultiClassEnvironmentHistory(env_init()).clairvoyant_rewards(bids, prices, T)

    results = [res_gpucb_known, res_gpucb_unknown, res_gpucb_one]
    titles = ["GP-UCB - Known classes", "GP-UCB - Unknown classes with context generation",
              "GP-UCB - Unknown classes using one context"]
    plot_multiple_single_class_results(results, opt_rewards, titles, True, '../Plots/step4_ucb_std.png')
    plot_multiple_single_class_results(results, opt_rewards, titles, False, '../Plots/step4_ucb.png')

    results = [res_gpucb_known, res_gpucb_unknown]
    titles = ["GP-UCB - Known classes", "GP-UCB - Unknown classes with context generation",]
    plot_multiple_single_class_results(results, opt_rewards, titles, True, '../Plots/step4_ucb_2.png')


if __name__ == '__main__':
    main()
