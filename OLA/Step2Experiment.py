import matplotlib.pyplot as plt
import numpy as np
import sklearn.gaussian_process.kernels
import warnings
from typing import Union
from datetime import datetime

from OLA.environments import SingleClassEnvironment, SingleClassEnvironmentHistory
from OLA.base_learners import Step2UCBLearner, Step2TSLearner
import new_environment_properties as ep
from OLA.simulators import simulate_single_class
from OLA.simulators import plot_single_class_sim_result
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


def env_init_step2(rng: np.random.Generator):
    N = ep.daily_clicks_curve
    en = lambda: ep.daily_click_curve_noise(rng, None)
    C = ep.click_cumulative_cost
    ec = lambda: ep.advertising_costs_curve_noise(rng, None)
    A = ep.click_conversion_rate

    return SingleClassEnvironment(N, en, C, ec, A, ep.get_production_cost(), rng)


def gpucb_learner_init(env: SingleClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                        kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float):
    return Step2UCBLearner(env, bids, prices, kernel, alpha, beta)


def gpts_learner_init(env: SingleClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                      kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, rng: np.random.Generator):
    return Step2TSLearner(env, bids, prices, kernel, alpha, rng)


def store_gp_mean_estimation(learner: Union[Step2TSLearner, Step2UCBLearner], n_estimations: list, c_estimations: list):
    if isinstance(learner, Step2UCBLearner):
        n_est = learner.n_estimator
        c_est = learner.c_estimator
    else:
        n_est = learner.n_estimator
        c_est = learner.c_estimator
    # now, the estimator are of different types (but inherit from the same parent)
    # and the two classes may call them with different names
    # practically they have the same name, so the previous "if" seems non-necessarily
    # n_est, c_est here are considered of type BaseGPEstimator
    n_estimations.append(n_est.mu_vector)
    c_estimations.append(c_est.mu_vector)


def plot_gp_mean_estimations(n_estimations: list, c_estimations: list):
    n_estimations = np.array(n_estimations)
    n_estimations = np.mean(n_estimations, axis=0)
    c_estimations = np.array(c_estimations)
    c_estimations = np.mean(c_estimations, axis=0)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 9))
    axes = axes.flatten()
    axes[0].plot(ep.get_bids(), n_estimations, label="estimation")
    axes[0].plot(ep.get_bids(), ep.daily_clicks_curve(ep.get_bids()), label="true value")
    axes[0].set_title("Number of clicks GP estimation")
    axes[0].legend()
    axes[1].plot(ep.get_bids(), c_estimations, label="estimation")
    axes[1].plot(ep.get_bids(), ep.click_cumulative_cost(ep.get_bids()), label="true value")
    axes[1].set_title("Advertising costs GP estimation")
    axes[1].legend()

    plt.show()


if __name__ == '__main__':
    seed = 1000
    bids = ep.get_bids()
    prices = ep.get_prices()
    rng = np.random.default_rng(seed=seed)
    env_init = lambda: env_init_step2(rng)

    # kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
    alpha = 10
    # beta should be around 110
    beta = 110

    learner_init_gpucb = lambda env, bids, prices: gpucb_learner_init(env, bids, prices, kernel, alpha, beta)
    learner_init_gpts = lambda env, bids, prices: gpts_learner_init(env, bids, prices, kernel, alpha, rng)

    warnings.filterwarnings("ignore")

    T = 365
    opt_rewards = SingleClassEnvironmentHistory(env_init_step2(rng)).clairvoyant_rewards(bids, prices, T)

    n_runs = 100
    click_estimations = []
    costs_estimations = []
    print("----GP-UCB----")
    start_time = datetime.now()
    sim_object_gpucb = simulate_single_class(env_init, bids, prices, learner_init_gpucb, T, n_runs=n_runs,
                                             hook_function=lambda learner: store_gp_mean_estimation(learner, click_estimations, costs_estimations))
    print("Elapsed time: ", (datetime.now() - start_time).total_seconds())
    plot_single_class_sim_result(sim_object_gpucb, opt_rewards)
    plot_gp_mean_estimations(click_estimations, costs_estimations)

    click_estimations = []
    costs_estimations = []
    print("----GP-TS----")
    start_time = datetime.now()
    sim_object_ts = simulate_single_class(env_init, bids, prices, learner_init_gpts, T, n_runs=n_runs,
                                          hook_function=lambda learner: store_gp_mean_estimation(learner, click_estimations, costs_estimations))
    print("Elapsed time: ", (datetime.now() - start_time).total_seconds())
    plot_single_class_sim_result(sim_object_ts, opt_rewards)
    plot_gp_mean_estimations(click_estimations, costs_estimations)
