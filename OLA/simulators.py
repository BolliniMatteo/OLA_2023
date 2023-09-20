# import inline as inline
import matplotlib

from environments import SingleClassEnvironment
from environments import SingleClassEnvironmentHistory
from environments import MultiClassEnvironment
from environments import MultiClassEnvironmentHistory
from base_learners import SingleClassLearner
from base_learners import MultiClassLearner

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config.plot_styling import *
from typing import Callable


class SingleClassSimResult:
    def __init__(self, inst_rewards_mean, inst_rewards_std,
                 inst_regrets_mean, inst_regrets_std,
                 cum_rewards_mean, cum_rewards_std,
                 cum_regrets_mean, cum_regrets_std):
        self.cum_rewards_mean = cum_rewards_mean
        self.cum_regrets_mean = cum_regrets_mean
        self.cum_rewards_std = cum_rewards_std
        self.cum_regrets_std = cum_regrets_std
        self.inst_regrets_mean = inst_regrets_mean
        self.inst_regrets_std = inst_regrets_std
        self.inst_rewards_std = inst_rewards_std
        self.inst_rewards_mean = inst_rewards_mean


def simulate_single_class(env_init: Callable[[], SingleClassEnvironment],
                          bids: np.ndarray, prices: np.ndarray,
                          learner_init: Callable[[SingleClassEnvironment, np.ndarray, np.ndarray], SingleClassLearner],
                          t: int, n_runs=300):
    """
    Execute n_runs simulations of a single class agent
    and returns the mean and std dev of the reward stats.
    We don't have reset functions for the agents and the environments, so I ask for the initialization ones
    :param env_init: a function without parameters that creates a new environment
    :param bids: the available bids
    :param prices: the available prices
    :param learner_init: a function that takes the environment, the bids and the prices, and creates a new single class learner
    :param t: the time horizon for each episode
    :param n_runs: the number of simulations to perform
    :return:
    """
    inst_rewards = []
    inst_regrets = []
    cum_rewards = []
    cum_regrets = []
    for i in range(n_runs):
        print('Experiment %d' % i)
        env = env_init()
        learner = learner_init(env, bids, prices)
        for j in range(t):
            # print('Iteration %d of experiment %d' % (j, i))
            learner.play_round()
        rewards, regrets, c_rewards, c_regrets = learner.history.reward_stats(bids, prices)
        inst_rewards.append(rewards)
        inst_regrets.append(regrets)
        cum_rewards.append(c_rewards)
        cum_regrets.append(c_regrets)
    inst_rewards = np.array(inst_rewards)
    inst_regrets = np.array(inst_regrets)
    cum_rewards = np.array(cum_rewards)
    cum_regrets = np.array(cum_regrets)

    inst_rewards_mean = np.mean(inst_rewards, axis=0)
    inst_regrets_mean = np.mean(inst_regrets, axis=0)
    cum_rewards_mean = np.mean(cum_rewards, axis=0)
    cum_regrets_mean = np.mean(cum_regrets, axis=0)

    inst_rewards_std = np.std(inst_rewards, axis=0)
    inst_regrets_std = np.std(inst_regrets, axis=0)
    cum_rewards_std = np.std(cum_rewards, axis=0)
    cum_regrets_std = np.std(cum_regrets, axis=0)

    return SingleClassSimResult(inst_rewards_mean, inst_rewards_std,
                                inst_regrets_mean, inst_regrets_std,
                                cum_rewards_mean, cum_rewards_std,
                                cum_regrets_mean, cum_regrets_std)


def plot_single_class_sim_result(result: SingleClassSimResult):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(50, 100))
    axes = axes.flatten()
    time_steps = [i for i in range(1, result.inst_regrets_mean.shape[0]+1)]
    axes[0].set_title('Instantaneous rewards')
    axes[0].plot(time_steps, result.inst_rewards_mean, label='mean')
    axes[0].plot(time_steps, result.inst_rewards_std, label='std')
    axes[0].legend()

    axes[1].set_title('Instantaneous regrets')
    axes[1].plot(time_steps, result.inst_regrets_mean, label='mean')
    axes[1].plot(time_steps, result.inst_regrets_std, label='std')
    axes[1].legend()

    axes[2].set_title('Cumulative reward')
    axes[2].plot(time_steps, result.cum_rewards_mean, label='mean')
    axes[2].plot(time_steps, result.cum_rewards_std, label='std')
    axes[2].legend()

    axes[3].set_title('Cumulative regret')
    axes[3].plot(time_steps, result.cum_regrets_mean, label='mean')
    axes[3].plot(time_steps, result.cum_regrets_std, label='std')
    axes[3].legend()

    plt.show()
