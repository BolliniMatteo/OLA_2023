import numpy as np
import matplotlib.pyplot as plt

from configs.plot_styling import *
from SingleClassBiddingEnvironment import *
from GPUCBLearner import *
from GPTSLearner import *


def fun(x):
    """
    Concave curve to learn, range of interest [1, 3]
    :param x: bid
    :return: number of clicks
    """
    return -3 * (x ** 2) + 12 * x - 7


if __name__ == '__main__':
    n_arms = 20
    min_bid = 1.0
    max_bid = 3.0
    bids = np.linspace(min_bid, max_bid, n_arms)
    # We make it very noisy
    sigma = 10

    T = 100
    n_experiments = 300
    gpucb_rewards_per_experiment = []
    gpts_rewards_per_experiment = []
    # In this case it is not necessary to reset the environment
    # at every experiment
    environment = SingleClassBiddingEnvironment(bids=bids, sigma=sigma, fun=fun)

    for e in range(0, n_experiments):
        gpucb_learner = GPUCBLearner(n_arms=n_arms, arms=bids)
        gpts_learner = GPTSLearner(n_arms=n_arms, arms=bids)

        for t in range(0, T):
            # GP-UCB learner
            print('Experiment %d round t %d' % (e, t))
            pulled_arm = gpucb_learner.pull_arm()
            reward = environment.round(pulled_arm)
            gpucb_learner.update(pulled_arm, reward)

            # GP-TS learner
            pulled_arm = gpts_learner.pull_arm()
            reward = environment.round(pulled_arm)
            gpts_learner.update(pulled_arm, reward)

        gpucb_rewards_per_experiment.append(gpucb_learner.collected_rewards)
        gpts_rewards_per_experiment.append(gpts_learner.collected_rewards)

    # Optimal reward on average
    optimal_reward = np.max(environment.means)
    plt.figure(0)
    plt.ylabel('Regret')
    plt.xlabel('t')
    plt.plot(np.cumsum(np.mean(optimal_reward - gpucb_rewards_per_experiment, axis=0)))
    plt.plot(np.cumsum(np.mean(optimal_reward - gpts_rewards_per_experiment, axis=0)))
    plt.legend(['GP-UCB', 'GP-TS'])
    plt.show()
