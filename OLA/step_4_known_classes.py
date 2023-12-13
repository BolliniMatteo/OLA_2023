# I am trying to understand why the learner with known classes performs badly

import numpy as np
import sklearn.gaussian_process.kernels
import warnings
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import new_environment_properties as ep
from OLA.multi_class_learners import Step4TSContextGenLearner, Step4TSRealClassesLearner, Step4TSOneClassLearner, \
    Step4UCBContextGenLearner, Step4UCBRealClassesLearner, Step4UCBOneClassLearner
from OLA.context_gen import ContextGenerator
from OLA.environments import MultiClassEnvironment, MultiClassEnvironmentHistory
from OLA.simulators import simulate_multi_class, plot_multi_class_sim_result, plot_multiple_single_class_results
from OLA.test_learners import Step4ClairvoyantLearner

warnings.filterwarnings("ignore")


def env_init_step4(rng: np.random.Generator):
    n_features = ep.n_features
    class_map = ep.class_map
    user_prob_map = ep.user_prob_map
    n = ep.daily_clicks_curve_multiclass
    en = lambda: ep.daily_click_curve_noise(rng, 1)
    c = ep.click_cumulative_cost_multiclass
    ec = lambda: ep.advertising_costs_curve_noise(rng, 1)
    a = ep.click_conversion_rate_multiclass
    return MultiClassEnvironment(n_features, class_map, user_prob_map, n, en, c, ec, a, ep.get_production_cost(), rng)


def gpucb_known_learner_init(env: MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                             kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float, c: float):
    return Step4UCBRealClassesLearner(env, bids, prices, kernel, alpha, beta, c)


def main():
    seed = 5000
    bids = ep.get_bids()
    prices = ep.get_prices()
    rng = np.random.default_rng(seed=seed)
    env_init = lambda: env_init_step4(rng)

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
    alpha = 10
    c = 0.2
    T = 365  # TODO: should be 365

    learner_init_gpts_known = lambda env: gpucb_known_learner_init(env, bids, prices, kernel, alpha, rng, c)

    n_runs = 10

    print("GP-TS learner, known classes", flush=True)
    res_ts_known = simulate_multi_class(env_init, learner_init_gpts_known, T, n_runs).aggregate_results

    opt_rewards = MultiClassEnvironmentHistory(env_init()).clairvoyant_rewards(bids, prices, T)

    plot_multiple_single_class_results([res_ts_known], opt_rewards, ["TS known classes"])


if __name__ == '__main__':
    main()
