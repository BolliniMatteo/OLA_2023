import numpy as np
import sklearn.gaussian_process.kernels

from OLA.environments import SingleClassEnvironment
from OLA.base_learners import Step3UCBLearner
from OLA.base_learners import Step3TSLearner
import environment_properties as ep
from OLA.simulators import simulate_single_class
from OLA.simulators import plot_single_class_sim_result
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


def env_init_step3(prices: np.ndarray, rng: np.random.Generator):
    N = ep.daily_clicks_curve
    en = lambda: ep.click_curve_noise(rng, 1)
    C = ep.click_cumulative_cost
    ec = lambda: ep.click_cumulative_cost_noise(rng, 1)
    A = {p: ep.click_conversion_rate(p) for p in prices}

    return SingleClassEnvironment(N, en, C, ec, A, rng)


def gpucb_learner_init(env: SingleClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                       kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float):
    return Step3UCBLearner(env, bids, prices, kernel, alpha, beta)


def gpts_learner_init(env: SingleClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                      kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, rng: np.random.Generator):
    return Step3TSLearner(env, bids, prices, kernel, alpha, rng)


if __name__ == '__main__':
    seed = 5000
    bids = ep.get_bids()
    prices = ep.get_prices()
    rng = np.random.default_rng(seed=seed)
    env_init = lambda: env_init_step3(prices, rng)

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
    alpha = 10
    # beta should be around 110
    beta = 110

    learner_init_gpucb = lambda env: gpucb_learner_init(env, bids, prices, kernel, alpha, beta)
    learner_init_gpts = lambda env: gpts_learner_init(env, bids, prices, kernel, alpha, rng)

    T = 365
    #sim_object_gpucb = simulate_single_class(env_init, learner_init_gpucb, T, n_runs=20)
    sim_object_ts = simulate_single_class(env_init, learner_init_gpts, T, n_runs=20)
    #plot_single_class_sim_result(sim_object_gpucb)
    plot_single_class_sim_result(sim_object_ts)
