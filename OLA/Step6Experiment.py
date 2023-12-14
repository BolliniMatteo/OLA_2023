import numpy as np

from OLA.environments import SingleClassEnvironmentNonStationary, SingleClassEnvironmentNonStationaryHistory
from OLA.base_learners import Step5UCBLearner, Step5UCBChangeDetectorLearner, Step6EXP3Learner
from OLA.base_learners import Step5UCBWINLearner
import new_environment_properties as ep
from OLA.simulators import simulate_single_class
from OLA.simulators import plot_single_class_sim_result, plot_multiple_single_class_results


def env_init_step6(rng: np.random.Generator):
    N = ep.daily_clicks_curve
    en = lambda: ep.daily_click_curve_noise(rng, None) # size None returns a scalar, size 1 an array of a single element
    C = ep.click_cumulative_cost
    ec = lambda: ep.advertising_costs_curve_noise(rng, None)
    A = ep.conversion_rate_high_frequency_phases

    return SingleClassEnvironmentNonStationary(N, en, C, ec, A, ep.get_production_cost(), rng)

if __name__ == '__main__':
    seed = 1000
    bids = np.linspace(1, 10, num=1)
    prices = ep.get_prices()
    rng = np.random.default_rng(seed=seed)
    T = 365
    n_runs = 40
    c = 0.2
    opt_rewards = SingleClassEnvironmentNonStationaryHistory(env_init_step6(rng)).clairvoyant_rewards(bids, prices, T)
    print("-----UCB-SW-----")
    sim_object_ucbsw = simulate_single_class(lambda: env_init_step6(rng),
                                            bids, prices,
                                            lambda env, bids, prices: Step5UCBWINLearner(env, bids, prices, 30, 5),
                                            T, n_runs=n_runs)
    print("-----UCB-CD-----")
    sim_object_ucbcd = simulate_single_class(lambda: env_init_step6(rng),
                                            bids, prices,
                                            lambda env, bids, prices: Step5UCBChangeDetectorLearner(env
                                                                                                    , bids
                                                                                                    , prices
                                                                                                    , c
                                                                                                    , 30),
                                            T, n_runs=n_runs)
    print("-----EXP3-----")
    sim_object_exp3 = simulate_single_class(lambda: env_init_step6(rng),
                                            bids, prices,
                                            lambda env, bids, prices: Step6EXP3Learner(env
                                                                                       , bids
                                                                                       , prices)
                                            , T
                                            , n_runs=n_runs)
    plot_multiple_single_class_results([sim_object_ucbsw, sim_object_ucbcd, sim_object_exp3]
                                       , opt_rewards
                                       , ['UCB-SW', 'UCB-CD', 'EXP3'])

