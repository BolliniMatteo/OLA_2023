import numpy as np
import sklearn.gaussian_process.kernels
import warnings
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import environment_properties as ep
from OLA.base_learners import Step4TSContextGenLearner, Step4TSOneClassLearner, Step4TSRealClassesLearner, \
    Step4UCBContextGenLearner, Step4UCBOneClassLearner, Step4UCBRealClassesLearner
from OLA.context_gen import ContextGeneration
from OLA.environments import MultiClassEnvironment
from OLA.simulators import simulate_multi_class, plot_multi_class_sim_result


# TODO: address the kernel warnings
warnings.filterwarnings("ignore")


def env_init_step4(rng: np.random.Generator):
    n_features = ep.n_features
    n_classes = ep.n_classes
    class_map = ep.class_map
    user_prob_map = ep.user_prob_map
    n = [lambda bid: ep.daily_clicks_curve_multiclass(bid, cl) for cl in range(n_classes)]
    en = lambda: ep.click_curve_noise(rng, 1)
    c = [lambda bid: ep.click_cumulative_cost_multiclass(bid, cl) for cl in range(n_classes)]
    ec = lambda: ep.click_cumulative_cost_noise(rng, 1)
    a = [lambda p: ep.click_conversion_rate_multiclass(p, cl) for cl in range(n_classes)]
    return MultiClassEnvironment(n_features, class_map, user_prob_map, n, en, c, ec, a, rng)


def gpucb_known_learner_init(env: MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                             kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float, burn_in: int):
    return Step4UCBRealClassesLearner(env, bids, prices, kernel, alpha, beta, burn_in)


def gpucb_unknown_learner_init(env: MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                               kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float,
                               context_gen: ContextGeneration, burn_in: int):
    return Step4UCBContextGenLearner(env, bids, prices, kernel, alpha, beta, context_gen, burn_in)


def gpucb_one_learner_init(env: MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                           kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float, burn_in: int):
    return Step4UCBOneClassLearner(env, bids, prices, kernel, alpha, beta, burn_in)


def gpts_known_learner_init(env: MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                            kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, rng: np.random.Generator,
                            burn_in: int):
    return Step4TSRealClassesLearner(env, bids, prices, kernel, alpha, rng, burn_in)


def gpts_unknown_learner_init(env: MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                              kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, rng: np.random.Generator,
                              context_gen: ContextGeneration, burn_in: int):
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

    learner_init_gpucb_known = lambda env: gpucb_known_learner_init(env, bids, prices, kernel, alpha, beta, burn_in)
    learner_init_gpucb_unknown = lambda env, context_gen: gpucb_unknown_learner_init(env, bids, prices, kernel, alpha,
                                                                                     beta, context_gen, burn_in)
    learner_init_gpucb_one = lambda env: gpucb_one_learner_init(env, bids, prices, kernel, alpha, beta, burn_in)

    learner_init_gpts_known = lambda env: gpts_known_learner_init(env, bids, prices, kernel, alpha, rng, burn_in)
    learner_init_gpts_unknown = lambda env, context_gen: gpts_unknown_learner_init(env, bids, prices, kernel, alpha,
                                                                                   rng, context_gen, burn_in)
    learner_init_gpts_one = lambda env: gpts_one_learner_init(env, bids, prices, kernel, alpha, rng, burn_in)

    T = 50  # TODO: should be 365
    n_runs = 1

    print("GP-UCB learner, known classes", flush=True)
    res_gpucb_known = simulate_multi_class(env_init, learner_init_gpucb_known, T, n_runs)

    # print("GP-UCB learner, unknown classes with context generation", flush=True)
    # res_gpucb_unknown = simulate_multi_class(env_init, learner_init_gpucb_unknown, T, n_runs)

    # print("GP-UCB learner, unknown classes using one context", flush=True)
    # res_gpucb_one = simulate_multi_class(env_init, learner_init_gpucb_one, T, n_runs)

    print("GP-TS learner, known classes", flush=True)
    res_ts_known = simulate_multi_class(env_init, learner_init_gpts_known, T, n_runs)

    # print("GP-TS learner, unknown classes with context generation", flush=True)
    # res_ts_unknown = simulate_multi_class(env_init, learner_init_gpts_unknown, T, n_runs)

    # print("GP-TS learner, unknown classes using one context", flush=True)
    # res_ts_one = simulate_multi_class(env_init, learner_init_gpts_one, T, n_runs)

    plot_multi_class_sim_result(res_gpucb_known)
    # plot_multi_class_sim_result(res_gpucb_unknown)
    # plot_multi_class_sim_result(res_gpucb_one)

    plot_multi_class_sim_result(res_ts_known)
    # plot_multi_class_sim_result(res_ts_unknown)
    # plot_multi_class_sim_result(res_ts_one)


if __name__ == '__main__':
    main()
