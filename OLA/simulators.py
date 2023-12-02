import sys
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from base_learners import SingleClassLearner, SingleClassLearnerNonStationary
from base_learners import MultiClassLearner
from environments import SingleClassEnvironment, MultiClassEnvironment, SingleClassEnvironmentNonStationary


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


def simulate_single_class(env_init: Callable[[], Union[SingleClassEnvironment, SingleClassEnvironmentNonStationary]],
                          bids: np.ndarray, prices: np.ndarray,
                          learner_init: Union[
                              Callable[[SingleClassEnvironment, np.ndarray, np.ndarray], SingleClassLearner],
                              Callable[[SingleClassEnvironmentNonStationary, np.ndarray,
                                        np.ndarray], SingleClassLearnerNonStationary]
                          ],
                          t: int, n_runs=300,
                          hook_function: Union[Callable[[SingleClassLearner], None], Callable[
                              [SingleClassLearnerNonStationary], None]] = None):
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
    :param hook_function: a function that will be called at the end of each experiment,
    useful to store any other data that you need to analyze regarding the agent performance
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
        if hook_function is not None:
            hook_function(learner)
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


def _plot_single_class_sim_result(result: SingleClassSimResult, opt_rewards: np.ndarray, fig, axes):
    time_steps = [i for i in range(1, result.inst_regrets_mean.shape[0] + 1)]
    axes[0].set_title('Instantaneous rewards')
    axes[0].plot(time_steps, result.inst_rewards_mean, label='mean')
    # axes[0].plot(time_steps, result.inst_rewards_mean + result.inst_rewards_std, label='std')
    axes[0].plot(time_steps, opt_rewards, label='optimal')
    axes[0].legend()

    axes[1].set_title('Instantaneous regrets')
    axes[1].plot(time_steps, result.inst_regrets_mean, label='mean')
    # axes[1].plot(time_steps, result.inst_regrets_mean + result.inst_regrets_std, label='std')
    axes[1].legend()

    axes[2].set_title('Cumulative reward')
    axes[2].plot(time_steps, result.cum_rewards_mean, label='mean')
    # axes[2].plot(time_steps, result.cum_rewards_mean + result.cum_rewards_std, label='std')
    axes[2].plot(time_steps, np.cumsum(opt_rewards), label='optimal')
    axes[2].legend()

    axes[3].set_title('Cumulative regret')
    axes[3].plot(time_steps, result.cum_regrets_mean, label='mean')
    # axes[3].plot(time_steps, result.cum_regrets_mean + result.cum_regrets_std, label='std')
    axes[3].legend()


def plot_single_class_sim_result(result: SingleClassSimResult, opt_rewards: np.ndarray, title: str = ""):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    axes = axes.flatten()
    fig.suptitle(title)
    _plot_single_class_sim_result(result, opt_rewards, fig, axes)
    plt.show()


def plot_multiple_single_class_results(results: list, opt_rewards: np.ndarray, algorithms: list, plot_std=False):
    # useful, for instance, to compare UCB and TS
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    axes = axes.flatten()
    time_steps = [i for i in range(1, opt_rewards.shape[0] + 1)]
    colors = ['b', 'r', 'c', 'm']
    for i in range(len(results)):
        axes[0].plot(time_steps, results[i].inst_rewards_mean, label=algorithms[i], color=colors[i])
        axes[1].plot(time_steps, results[i].inst_regrets_mean, label=algorithms[i], color=colors[i])
        axes[2].plot(time_steps, results[i].cum_rewards_mean, label=algorithms[i], color=colors[i])
        axes[3].plot(time_steps, results[i].cum_regrets_mean, label=algorithms[i], color=colors[i])

        if plot_std is True:
            axes[0].plot(time_steps, results[i].inst_rewards_mean + results[i].inst_rewards_std,
                         linestyle='dashed', color=colors[i])
            axes[0].plot(time_steps, results[i].inst_rewards_mean - results[i].inst_rewards_std,
                         linestyle='dashed', color=colors[i])
            axes[1].plot(time_steps, results[i].inst_regrets_mean + results[i].inst_regrets_std,
                         linestyle='dashed', color=colors[i])
            axes[1].plot(time_steps, results[i].inst_regrets_mean - results[i].inst_regrets_std,
                         linestyle='dashed', color=colors[i])
            axes[2].plot(time_steps, results[i].cum_rewards_mean + results[i].cum_rewards_std,
                         linestyle='dashed', color=colors[i])
            axes[2].plot(time_steps, results[i].cum_rewards_mean - results[i].cum_rewards_std,
                         linestyle='dashed', color=colors[i])
            axes[3].plot(time_steps, results[i].cum_regrets_mean + results[i].cum_regrets_std,
                         linestyle='dashed', color=colors[i])
            axes[3].plot(time_steps, results[i].cum_regrets_mean - results[i].cum_regrets_std,
                         linestyle='dashed', color=colors[i])

    axes[0].plot(time_steps, opt_rewards, label='optimal', color='g')
    axes[2].plot(time_steps, np.cumsum(opt_rewards), label='optimal', color='g')
    axes[0].set_title('Instantaneous rewards')
    axes[1].set_title('Instantaneous regrets')
    axes[2].set_title('Cumulative reward')
    axes[3].set_title('Cumulative regret')
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    axes[3].legend()
    plt.show()


def plot_multi_class_sim_result(result: MultiClassSimResult, opt_rewards, title: str = ""):
    # TODO: this at the moment only plots aggregate results
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    axes = axes.flatten()
    fig.suptitle(title)
    _plot_single_class_sim_result(result.aggregate_results, opt_rewards, fig, axes)
    # if title:
    #     filename = f"{title}.png"
    #     print(f"Saving {filename}")
    #     plt.savefig(filename)
    plt.show()
