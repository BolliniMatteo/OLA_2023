import numpy as np
import sklearn.gaussian_process.kernels
import warnings
from datetime import datetime

from OLA.environments import SingleClassEnvironment, SingleClassEnvironmentHistory
from OLA.base_learners import Step3UCBLearner
from OLA.base_learners import Step3TSLearner
import new_environment_properties as ep
from OLA.simulators import simulate_single_class
from OLA.simulators import plot_single_class_sim_result, plot_multiple_single_class_results
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


def env_init_step3(rng: np.random.Generator):
    N = ep.daily_clicks_curve
    en = lambda: ep.daily_click_curve_noise(rng, 1)
    C = ep.click_cumulative_cost
    ec = lambda: ep.advertising_costs_curve_noise(rng, 1)
    A = ep.click_conversion_rate

    return SingleClassEnvironment(N, en, C, ec, A, ep.get_production_cost(), rng)


def gpucb_learner_init(env: SingleClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                       kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float, c: float):
    return Step3UCBLearner(env, bids, prices, kernel, alpha, beta, c)


def gpts_learner_init(env: SingleClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                      kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, rng: np.random.Generator):
    return Step3TSLearner(env, bids, prices, kernel, alpha, rng)


if __name__ == '__main__':
    seed = 5000
    bids = ep.get_bids()
    prices = ep.get_prices()
    rng = np.random.default_rng(seed=seed)
    env_init = lambda: env_init_step3(rng)

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
    alpha = 10
    # beta should be around 110
    beta = 110

    c = 0.2

    learner_init_gpucb = lambda env, bids, prices: gpucb_learner_init(env, bids, prices, kernel, alpha, beta, c)
    learner_init_gpts = lambda env, bids, prices: gpts_learner_init(env, bids, prices, kernel, alpha, rng)

    warnings.filterwarnings("ignore")

    T = 365
    opt_rewards = SingleClassEnvironmentHistory(env_init_step3(rng)).clairvoyant_rewards(bids, prices, T)
    n_runs = 100

    print("----UCB----")
    start_time = datetime.now()
    sim_object_gpucb = simulate_single_class(env_init, bids, prices, learner_init_gpucb, T, n_runs=n_runs)
    print("Elapsed time: ", (datetime.now()-start_time).total_seconds())
    # plot_single_class_sim_result(sim_object_gpucb, opt_rewards)

    print("----TS----")
    start_time = datetime.now()
    sim_object_ts = simulate_single_class(env_init, bids, prices, learner_init_gpts, T, n_runs=n_runs)
    print("Elapsed time: ", (datetime.now()-start_time).total_seconds())
    # plot_single_class_sim_result(sim_object_ts, opt_rewards)

    plot_multiple_single_class_results([sim_object_gpucb, sim_object_ts], opt_rewards, ['GP-UCB1', 'GP-TS'])
