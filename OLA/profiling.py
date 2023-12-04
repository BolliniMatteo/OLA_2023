import numpy as np
import sklearn.gaussian_process.kernels
import warnings
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import cProfile
import pstats
import re

import new_environment_properties as ep
from OLA.base_learners import Step4TSContextGenLearner, Step4TSOneClassLearner, Step4TSRealClassesLearner, \
    Step4UCBContextGenLearner, Step4UCBOneClassLearner, Step4UCBRealClassesLearner
from OLA.context_gen import ContextGeneration
from OLA.environments import MultiClassEnvironment, MultiClassEnvironmentHistory
from OLA.simulators import simulate_multi_class, plot_multi_class_sim_result
from OLA.test_learners import Step4ClairvoyantLearner

# TODO: address the kernel warnings
warnings.filterwarnings("ignore")


def env_init_step4(rng: np.random.Generator):
    n_features = ep.n_features
    n_classes = ep.n_classes
    class_map = ep.class_map
    user_prob_map = ep.user_prob_map
    n = [lambda bid: ep.daily_clicks_curve_multiclass(bid, cl) for cl in range(n_classes)]
    en = lambda: ep.daily_click_curve_noise(rng, 1)
    c = [lambda bid: ep.click_cumulative_cost_multiclass(bid, cl) for cl in range(n_classes)]
    ec = lambda: ep.advertising_costs_curve_noise(rng, 1)
    a = [lambda p: ep.click_conversion_rate_multiclass(p, cl) for cl in range(n_classes)]
    return MultiClassEnvironment(n_features, class_map, user_prob_map, n, en, c, ec, a, ep.get_production_cost(), rng)


def gpucb_known_learner_init(env: MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                             kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float, burn_in: int):
    return Step4UCBRealClassesLearner(env, bids, prices, kernel, alpha, beta, burn_in)


def gpucb_unknown_learner_init(env: MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                               kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float,
                               burn_in: int, rng: np.random.Generator, bound_confidence: float):
    context_gen = ContextGeneration(env, bids, prices, kernel, alpha, rng, beta, bound_confidence)
    return Step4UCBContextGenLearner(env, bids, prices, kernel, alpha, beta, context_gen, burn_in)


def gpucb_one_learner_init(env: MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                           kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float, burn_in: int):
    return Step4UCBOneClassLearner(env, bids, prices, kernel, alpha, beta, burn_in)


def gpts_known_learner_init(env: MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                            kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, rng: np.random.Generator,
                            burn_in: int):
    return Step4TSRealClassesLearner(env, bids, prices, kernel, alpha, rng, burn_in)


def gpts_unknown_learner_init(env: MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                              kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float,
                              rng: np.random.Generator, burn_in: int, bound_confidence: float):
    context_gen = ContextGeneration(env, bids, prices, kernel, alpha, rng, beta, bound_confidence)
    return Step4TSContextGenLearner(env, bids, prices, kernel, alpha, rng, context_gen, burn_in)


def gpts_one_learner_init(env: MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                          kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, rng: np.random.Generator,
                          burn_in: int):
    return Step4TSOneClassLearner(env, bids, prices, kernel, alpha, rng, burn_in)


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
    burn_in = 500
    hoeffding_bound_confidence = 0.95

    learner_init_gpucb_known = lambda env: gpucb_known_learner_init(env, bids, prices, kernel, alpha, beta, burn_in)
    learner_init_gpucb_unknown = lambda env: gpucb_unknown_learner_init(env, bids, prices, kernel, alpha, beta, burn_in,
                                                                        rng, hoeffding_bound_confidence)
    learner_init_gpucb_one = lambda env: gpucb_one_learner_init(env, bids, prices, kernel, alpha, beta, burn_in)

    learner_init_gpts_known = lambda env: gpts_known_learner_init(env, bids, prices, kernel, alpha, rng, burn_in)
    learner_init_gpts_unknown = lambda env: gpts_unknown_learner_init(env, bids, prices, kernel, alpha, beta, rng,
                                                                      burn_in, hoeffding_bound_confidence)
    learner_init_gpts_one = lambda env: gpts_one_learner_init(env, bids, prices, kernel, alpha, rng, burn_in)

    T = 365  # TODO: should be 365
    n_runs = 1

    # print("GP-UCB learner, known classes", flush=True)
    res_gpucb_known = simulate_multi_class(env_init, learner_init_gpucb_known, T, n_runs)

    # print("GP-UCB learner, unknown classes with context generation", flush=True)
    # res_gpucb_unknown = simulate_multi_class(env_init, learner_init_gpucb_unknown, T, n_runs)

    # print("GP-UCB learner, unknown classes using one context", flush=True)
    # res_gpucb_one = simulate_multi_class(env_init, learner_init_gpucb_one, T, n_runs)

    # print("GP-TS learner, known classes", flush=True)
    # res_ts_known = simulate_multi_class(env_init, learner_init_gpts_known, T, n_runs)

    # print("GP-TS learner, unknown classes with context generation", flush=True)
    # res_ts_unknown = simulate_multi_class(env_init, learner_init_gpts_unknown, T, n_runs)

    # print("GP-TS learner, unknown classes using one context", flush=True)
    # res_ts_one = simulate_multi_class(env_init, learner_init_gpts_one, T, n_runs)
    """
    opt_rewards = MultiClassEnvironmentHistory(env_init()).clairvoyant_rewards(bids, prices, T)

    plot_multi_class_sim_result(res_gpucb_known, opt_rewards, "GP-UCB - Known classes")
    
    plot_multi_class_sim_result(res_gpucb_unknown, opt_rewards, "GP-UCB - Unknown classes with context generation")
    plot_multi_class_sim_result(res_gpucb_one, opt_rewards, "GP-UCB - Unknown classes using one context")

    plot_multi_class_sim_result(res_ts_known, opt_rewards, "GP-TS - Known classes")
    plot_multi_class_sim_result(res_ts_unknown, opt_rewards, "GP-TS - Unknown classes with context generation")
    plot_multi_class_sim_result(res_ts_one, opt_rewards, "GP-TS - Unknown classes using one context")
    """


if __name__ == '__main__':
    with cProfile.Profile() as pr:
        pr.enable()
        main()
        pr.print_stats()
        stats = pstats.Stats(pr)
        stats.sort_stats('cumulative')
        # name of the files of the functions we want to see the stats of
        stats.print_stats('base_learners|environments|estimators')
        # it prints stats for every function
        # then at the end it reprints only the one we filtered

