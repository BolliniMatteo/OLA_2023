import sys
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from base_learners import SingleClassLearner
from base_learners import MultiClassLearner
from environments import SingleClassEnvironment
from environments import MultiClassEnvironment


class SingleClassSimResult:
    def __init__(self, inst_rewards_mean, inst_rewards_std,
                 inst_regrets_mean, inst_regrets_std,
                 cum_rewards_mean, cum_rewards_std,
                 cum_regrets_mean, cum_regrets_std):
        self.cum_rewards_mean = cum_rewards_mean
        self.cum_rewards_std = cum_rewards_std
        self.cum_regrets_mean = cum_regrets_mean
        self.cum_regrets_std = cum_regrets_std
        self.inst_regrets_mean = inst_regrets_mean
        self.inst_regrets_std = inst_regrets_std
        self.inst_rewards_mean = inst_rewards_mean
        self.inst_rewards_std = inst_rewards_std


class MultiClassSimResult:
    def __init__(self, per_class_results: list[SingleClassSimResult], aggregate_results: SingleClassSimResult):
        self.per_class_results = per_class_results
        self.aggregate_results = aggregate_results


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


def simulate_multi_class(env_init: Callable[[], MultiClassEnvironment],
                         learner_init: Callable[[MultiClassEnvironment], MultiClassLearner],
                         t: int, n_runs=300):
    n_classes = env_init().classes_count()

    inst_rewards = np.zeros((n_classes, n_runs, t))
    inst_regrets = np.zeros((n_classes, n_runs, t))
    cum_rewards = np.zeros((n_classes, n_runs, t))
    cum_regrets = np.zeros((n_classes, n_runs, t))

    aggr_inst_rewards = np.zeros((n_runs, t))
    aggr_inst_regrets = np.zeros((n_runs, t))
    aggr_cum_rewards = np.zeros((n_runs, t))
    aggr_cum_regrets = np.zeros((n_runs, t))

    for i in range(n_runs):
        env = env_init()
        learner = learner_init(env)

        for _ in tqdm(range(t), desc=f"Run #{i}", file=sys.stdout):
            learner.play_round()

        rewards, regrets, c_rewards, c_regrets = learner.history.stats_for_class(learner.xs, learner.ps)
        inst_rewards[:, i] = rewards
        inst_regrets[:, i] = regrets
        cum_rewards[:, i] = c_rewards
        cum_regrets[:, i] = c_regrets

        aggr_rewards, aggr_regrets, aggr_c_rewards, aggr_c_regrets = learner.history.stats_total(learner.xs, learner.ps)
        aggr_inst_rewards[i] = aggr_rewards
        aggr_inst_regrets[i] = aggr_regrets
        aggr_cum_rewards[i] = aggr_c_rewards
        aggr_cum_regrets[i] = aggr_c_regrets

    return MultiClassSimResult([
        SingleClassSimResult(
            np.mean(inst_rewards[cl], axis=0),
            np.std(inst_rewards[cl], axis=0),
            np.mean(inst_regrets[cl], axis=0),
            np.std(inst_regrets[cl], axis=0),
            np.mean(cum_rewards[cl], axis=0),
            np.std(cum_rewards[cl], axis=0),
            np.mean(cum_regrets[cl], axis=0),
            np.std(cum_regrets[cl], axis=0)
        ) for cl in range(n_classes)
    ], SingleClassSimResult(
        np.mean(aggr_inst_rewards, axis=0),
        np.std(aggr_inst_rewards, axis=0),
        np.mean(aggr_inst_regrets, axis=0),
        np.std(aggr_inst_regrets, axis=0),
        np.mean(aggr_cum_rewards, axis=0),
        np.std(aggr_cum_rewards, axis=0),
        np.mean(aggr_cum_regrets, axis=0),
        np.std(aggr_cum_regrets, axis=0)
    ))


def _plot_single_class_sim_result(result: SingleClassSimResult, fig, axes):
    time_steps = [i for i in range(1, result.inst_regrets_mean.shape[0]+1)]
    axes[0].set_title('Instantaneous rewards')
    axes[0].plot(time_steps, result.inst_rewards_mean, label='mean')
    # axes[0].plot(time_steps, result.inst_rewards_mean + result.inst_rewards_std, label='std')
    axes[0].legend()

    axes[1].set_title('Instantaneous regrets')
    axes[1].plot(time_steps, result.inst_regrets_mean, label='mean')
    # axes[1].plot(time_steps, result.inst_regrets_mean + result.inst_regrets_std, label='std')
    axes[1].legend()

    axes[2].set_title('Cumulative reward')
    axes[2].plot(time_steps, result.cum_rewards_mean, label='mean')
    # axes[2].plot(time_steps, result.cum_rewards_mean + result.cum_rewards_std, label='std')
    axes[2].legend()

    axes[3].set_title('Cumulative regret')
    axes[3].plot(time_steps, result.cum_regrets_mean, label='mean')
    # axes[3].plot(time_steps, result.cum_regrets_mean + result.cum_regrets_std, label='std')
    axes[3].legend()


def plot_single_class_sim_result(result: SingleClassSimResult, title: str = ""):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    axes = axes.flatten()
    fig.suptitle(title)
    _plot_single_class_sim_result(result, fig, axes)
    plt.show()


def plot_multi_class_sim_result(result: MultiClassSimResult, title: str = ""):
    # TODO: this at the moment only plots aggregate results
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    axes = axes.flatten()
    fig.suptitle(title)
    _plot_single_class_sim_result(result.aggregate_results, fig, axes)
    # if title:
    #     filename = f"{title}.png"
    #     print(f"Saving {filename}")
    #     plt.savefig(filename)
    plt.show()
