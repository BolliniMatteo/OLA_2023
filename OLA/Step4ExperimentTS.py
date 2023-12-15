import numpy as np
import sklearn.gaussian_process.kernels
import warnings
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import new_environment_properties as ep
from OLA.multi_class_learners import Step4TSRealClassesLearner, Step4TSOneClassLearner, Step4TSStatefulContextLearner
from OLA.context_gen import StatefulContextGenerator
from OLA.environments import MultiClassEnvironment, MultiClassEnvironmentHistory
from OLA.simulators import simulate_multi_class,  plot_multiple_single_class_results

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


def gpts_known_learner_init(env: MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                            kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, rng: np.random.Generator):
    return Step4TSRealClassesLearner(env, bids, prices, kernel, alpha, rng)


def gpts_unknown_learner_init(env: MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                              kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float,
                              rng: np.random.Generator, burn_in: int, bound_confidence: float):
    context_gen = StatefulContextGenerator(env, bids, prices, kernel, alpha, rng, beta, bound_confidence)
    return Step4TSStatefulContextLearner(env, bids, prices, kernel, alpha, rng, context_gen, burn_in)


def gpts_one_learner_init(env: MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                          kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, rng: np.random.Generator):
    return Step4TSOneClassLearner(env, bids, prices, kernel, alpha, rng)


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
    burn_in = 0 # should be 0
    T = 365  # should be 365
    hoeffding_bound_confidence = 0.90

    learner_init_gpts_known = lambda env: gpts_known_learner_init(env, bids, prices, kernel, alpha, rng)
    learner_init_gpts_unknown = lambda env: gpts_unknown_learner_init(env, bids, prices, kernel, alpha, beta, rng,
                                                                      burn_in, hoeffding_bound_confidence)
    learner_init_gpts_one = lambda env: gpts_one_learner_init(env, bids, prices, kernel, alpha, rng)

    n_runs = 100

    print("GP-TS learner, known classes", flush=True)
    res_ts_known = simulate_multi_class(env_init, learner_init_gpts_known, T, n_runs).aggregate_results

    print("GP-TS learner, unknown classes with context generation", flush=True)
    res_ts_unknown = simulate_multi_class(env_init, learner_init_gpts_unknown, T, n_runs).aggregate_results

    print("GP-TS learner, unknown classes using one context", flush=True)
    res_ts_one = simulate_multi_class(env_init, learner_init_gpts_one, T, n_runs).aggregate_results

    opt_rewards = MultiClassEnvironmentHistory(env_init()).clairvoyant_rewards(bids, prices, T)

    results = [res_ts_known, res_ts_unknown, res_ts_one]
    titles = ["GP-TS - Known classes", "GP-TS - Unknown classes with context generation",
              "GP-TS - Unknown classes using one context"]
    plot_multiple_single_class_results(results, opt_rewards, titles, True, '../Plots/step4_ts_std.png')
    plot_multiple_single_class_results(results, opt_rewards, titles, False, '../Plots/step4_ts.png')


if __name__ == '__main__':
    main()
