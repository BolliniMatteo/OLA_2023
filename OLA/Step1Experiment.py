import os

import numpy as np
from typing import Union

from OLA.environments import SingleClassEnvironment,SingleClassEnvironmentHistory
from OLA.base_learners import Step1UCBLearner
from OLA.base_learners import Step1TSLearner
import new_environment_properties as ep
from OLA.simulators import simulate_single_class, plot_multiple_single_class_results
from OLA.simulators import plot_single_class_sim_result


def env_init_step1(rng: np.random.Generator):
    N = ep.daily_clicks_curve
    en = lambda: ep.daily_click_curve_noise(rng, None) # size None returns a scalar, size 1 an array of a single element
    C = ep.click_cumulative_cost
    ec = lambda: ep.advertising_costs_curve_noise(rng, None)
    A = ep.click_conversion_rate

    return SingleClassEnvironment(N, en, C, ec, A, ep.get_production_cost(), rng)


class WorstArmCounter:
    def __init__(self, worst_arm):
        self.worst_arm = worst_arm
        self.count = 0

    def count_worst_arm(self, learner: Union[Step1TSLearner, Step1TSLearner]):
        ps = np.array(learner.history.ps)
        mask = ps == self.worst_arm
        self.count += np.sum(mask)


if __name__ == '__main__':
    seed = 2000
    bids = ep.get_bids()
    prices = ep.get_prices()
    rng = np.random.default_rng(seed=seed)
    T = 365
    n_runs = 100
    opt_rewards = SingleClassEnvironmentHistory(env_init_step1(rng)).clairvoyant_rewards(bids, prices, T)
    worst_arm_count_ucb = WorstArmCounter(prices[3])
    sim_object_ucb1 = simulate_single_class(lambda: env_init_step1(rng),
                                            bids, prices,
                                            lambda env, bids, prices: Step1UCBLearner(env, bids, prices, 0.4),
                                            T, n_runs,
                                            lambda learner: worst_arm_count_ucb.count_worst_arm(learner))

    worst_arm_count_ts = WorstArmCounter(prices[-1])
    sim_object_ts = simulate_single_class(lambda: env_init_step1(rng),
                                          bids, prices,
                                          lambda env, bids, prices: Step1TSLearner(env, bids, prices, rng),
                                          T, n_runs,
                                          lambda learner: worst_arm_count_ts.count_worst_arm(learner))

    print("UCB - number of times (on average) the worst arm is played: ", worst_arm_count_ucb.count / n_runs)
    print("TS - number of times (on average) the worst arm is played: ", worst_arm_count_ts.count / n_runs)
    plot_multiple_single_class_results([sim_object_ucb1, sim_object_ts], opt_rewards, ['UCB1', 'TS'],
                                       True, 'Plots/step1.png')


"""
Seed 1000
With 300 runsTS seems worst than the optimized UCB1
It's probably due to the worst arm that is very bad. With 300 runs
UCB plays it only once,
TS plays it 1.01 times... so there are a few learners that play it two times

UCB becomes worst if I increase to 1000 runs, where TS plays the worst arm on average 1.017 times
With 3000 runs, again UCB plays the worst arm 1 time, TS 1.014 times

(or simply change seed so that you don't get the unlucky run)

Now I changed the classes, this holds no more
"""