import math
import numpy as np
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from typing import Union

"""
I try to keep things modular
Estimator simply estimate things without optimizing
They receive data and produce estimations, but do not play

I try to keep them independent, so they keep their state
and do not rely on the environment history
"""


class BeUCB1SWEstimator:
    def __init__(self, na: int, window_size: int, c: float = 1):
        """
        :param na: the number of arms
        :param c: a constant multiplicative factor for the UCB bound.
        1 to use the default bound
        """
        self.na = na
        self.pulled_per_arm = []
        self.rewards_per_arm = []
        self.means = np.zeros(na, dtype=float)
        self.window_size = window_size
        self.c = c
        for _ in range(na):
            arm = []
            rewards = []
            self.pulled_per_arm.append(arm)
            self.rewards_per_arm.append(rewards)
        self.rounds = 0

    def get_non_pulled_arms(self):
        """
        :return: an array with the indices of the arms that result non-pulled,
        in the sense that the estimation for them have never been updated by update_estimation.
        An array is returned even in case of a single non-pulled arm.
        An empty array of shape (0,) is returned when there aren't any non-pulled arms
        """
        window_play_counts = self.get_window_play_counts()
        zero_mask = (window_play_counts == 0)
        indices = np.arange(self.means.shape[0])
        return indices[zero_mask]

    def get_window_play_counts(self):
        if self.rounds < self.window_size:
            window_play_counts = np.sum(np.array(self.pulled_per_arm), axis=1)
        else:
            window_play_counts = np.sum(np.array(self.pulled_per_arm)[:, -self.window_size:], axis=1)
        return window_play_counts

    def window_compute_means(self):
        if self.rounds < self.window_size:
            sum_pulls = np.sum(np.array(self.pulled_per_arm), axis=1)
            sum_rewards = np.sum(np.array(self.rewards_per_arm), axis=1)
            self.means[sum_pulls == 0] = 0
            self.means[sum_pulls > 0] = sum_rewards[sum_pulls > 0] / sum_pulls[sum_pulls > 0]
            # print(self.means)
        else:
            sum_pulls = np.sum(np.array(self.pulled_per_arm)[:, -self.window_size:], axis=1)
            sum_rewards = np.sum(np.array(self.rewards_per_arm)[:, -self.window_size:], axis=1)
            self.means[sum_pulls == 0] = 0
            self.means[sum_pulls > 0] = sum_rewards[sum_pulls > 0] / sum_pulls[sum_pulls > 0]
            # print(self.means)

    def provide_estimations(self):
        """
        Provides the estimation of the average rewards
        If some arms have not been pulled, it returns +- inf (depending on the lower_bound) for them
        :return: the estimations (np array)
        """
        window_play_counts = self.get_window_play_counts()
        t = np.sum(window_play_counts)
        zero_mask = (window_play_counts == 0)
        non_zero_mask = np.logical_not(zero_mask)
        thetas = np.copy(self.means)
        thetas[zero_mask] = float('+inf')
        thetas[non_zero_mask] = thetas[non_zero_mask] + self.c * np.sqrt((2 * np.log(t)) / window_play_counts)
        return thetas

    def update_estimations(self, played_arm: int, positive_rewards: int, total_rewards: int):
        """
        Updates the internal attributes for the estimations
        :param played_arm: the arm that has been played in the last round
        :param positive_rewards: the number of positive rewards (r=1) collected in the last round
        :param total_rewards: the total rewards collected in the last round
        :return: None
        """
        for i in range(self.na):
            if i == played_arm:
                self.pulled_per_arm[i].append(total_rewards)
                self.rewards_per_arm[i].append(positive_rewards)
            else:
                self.pulled_per_arm[i].append(0)
                self.rewards_per_arm[i].append(0)
        self.window_compute_means()
        self.rounds += 1


class BeUCB1Estimator:
    """
    An object that performs UCB1 on Bernoulli-like arms
    It simply provides the estimations (mean + UCB bound) and updates them
    Optimizing and playing are left to the client

    Arms are referred as indices from 0 to "number of arms - 1"
    You may want to use a dictionary, list or other sort of mappings with the true value
    """

    def __init__(self, na: int, c: float = 1):
        """
        :param na: the number of arms
        :param c: a constant multiplicative factor for the UCB bound.
        1 to use the default bound
        """
        self.na = na
        self.play_counts = np.zeros(na, dtype=int)
        self.means = np.zeros(na, dtype=float)
        self.c = c

    def reset_estimates(self):
        """
        This for when a change detection algorithm decides
        to reset everything
        """
        self.means = np.zeros(self.na, dtype=float)
        self.play_counts = np.zeros(self.na, dtype=int)

    def provide_estimations(self, lower_bound=False):
        """
        Provides the estimation of the average rewards
        If some arms have not been pulled, it returns +- inf (depending on the lower_bound) for them
        :param lower_bound: True if you want a lower bound, false otherwise
        :return: the estimations (np array)
        """
        t = np.sum(self.play_counts)
        zero_mask = (self.play_counts == 0)
        non_zero_mask = np.logical_not(zero_mask)
        thetas = np.copy(self.means)
        if lower_bound:
            thetas[zero_mask] = float('-inf')
            thetas[non_zero_mask] = thetas[non_zero_mask] - self.c * np.sqrt((2 * np.log(t)) / self.play_counts)
        else:
            thetas[zero_mask] = float('+inf')
            thetas[non_zero_mask] = thetas[non_zero_mask] + self.c * np.sqrt((2 * np.log(t)) / self.play_counts)
        return thetas

    def update_estimations(self, played_arm: int, positive_rewards: int, total_rewards: int):
        """
        Updates the internal attributes for the estimations
        :param played_arm: the arm that has been played in the last round
        :param positive_rewards: the number of positive rewards (r=1) collected in the last round
        :param total_rewards: the total rewards collected in the last round
        :return: None
        """
        self.means[played_arm] = self.means[played_arm] * self.play_counts[played_arm] + positive_rewards
        self.play_counts[played_arm] = self.play_counts[played_arm] + total_rewards
        self.means[played_arm] = self.means[played_arm] / self.play_counts[played_arm]

    def get_non_pulled_arms(self):
        """
        :return: an array with the indices of the arms that result non-pulled,
        in the sense that the estimation for them have never been updated by update_estimation.
        An array is returned even in case of a single non-pulled arm.
        An empty array of shape (0,) is returned when there aren't any non-pulled arms
        """
        zero_mask = (self.play_counts == 0)
        indices = np.arange(self.means.shape[0])
        return indices[zero_mask]


class BeTSEstimator:
    """
    An object that performs TS on Bernoulli-like arms
    It simply provides the estimations (samples from beta distributions) and updates them
    Optimizing and playing are left to the client

    Arms are referred as indices from 0 to "number of arms - 1"
    You may want to use a dictionary, list or other sort of mappings with the true value
    """

    def __init__(self, na: int, rng: np.random.Generator):
        """
        :param na: the number of arms
        :param rng: the numpy random generator that will be used to sample the betas
        """
        self.alphas = np.ones(na, dtype=int)
        self.betas = np.ones(na, dtype=int)
        self.rng = rng

    def provide_estimations(self):
        """
        Provides the estimation of the average rewards
        :return: the estimations (np array)
        """
        thetas = self.rng.beta(a=self.alphas, b=self.betas, size=self.alphas.shape)
        return thetas

    def update_estimations(self, played_arm: int, positive_rewards: int, total_rewards: int):
        """
        Updates the internal attributes for the estimations
        :param played_arm: the arm that has been played in the last round
        :param positive_rewards: the number of positive rewards (r=1) collected in the last round
        :param total_rewards: the total rewards collected in the last round
        :return: None
        """
        self.alphas[played_arm] += positive_rewards
        self.betas[played_arm] += total_rewards - positive_rewards


class BaseGPEstimator:
    """
    Base class for GP-UCB1 and GP-TS
    It keeps stored the means and standard deviations for each arm,
    and updates both them and the GP when new data arrives.
    When there are no data, the means are set to 0 and the standard deviations to 3 as default values
    """

    def __init__(self, arms: np.ndarray, kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float):
        """
        :param arms: the available arms (here arms are the true values, not their indices)
        :param kernel: the kernel to be used by the GP
        :param alpha: the alpha to be used by the GP
        """
        self.arms = arms
        # this is from the lab
        # self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=9)
        # this is from Carlos
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2, normalize_y=False)
        # this data could be found also in the environment history, but I prefer to keep this class independent
        self.played_arms = []
        self.rewards = []
        self.mu_vector = np.zeros(self.arms.shape[0])
        # We choose a standard deviation of 3 (???)
        self.sigma_vector = np.ones(self.arms.shape[0]) * 3

    def _update_gp(self):
        x = np.array(self.played_arms)
        X = np.reshape(x, (x.shape[0], 1))
        y = np.array(self.rewards)
        self.gp.fit(X, y)
        self.mu_vector, self.sigma_vector = self.gp.predict(np.atleast_2d(self.arms).T,
                                                            return_std=True)
        # depending on the np version, it may return a (100,1) or (100,) array
        # we need the latter
        self.mu_vector = np.reshape(self.mu_vector, self.arms.shape)
        self.sigma_vector = np.reshape(self.sigma_vector, self.arms.shape)

    def update_model(self, played_arms: Union[int, list, np.ndarray], rewards: Union[float, list, np.ndarray]):
        """
        Updates the internal attributes for the estimations
        :param played_arms: the arm(s) that have been played
        :param rewards: the reward(s) obtained in the corresponding round(s)
        Reward here is the target to be estimated, not necessarily the complete reward that the agent gets
        :return: None
        """
        if isinstance(played_arms, list):
            self.played_arms += list(played_arms)
            self.rewards += list(rewards)
        elif isinstance(played_arms, np.ndarray):
            self.played_arms += played_arms.tolist()
            self.rewards += rewards.tolist()
        else:
            self.played_arms.append(played_arms)
            self.rewards.append(rewards)
        self._update_gp()


class BeEXP3Estimator:
    def __init__(self, prices: np.ndarray, rng: np.random.Generator, gamma=0.4):
        self.gamma = gamma
        self.weights = np.full(prices.shape[0], 1.0)
        self.arm_prices = prices
        self.probability_distribution = None
        self.rng = rng
        self.update_distributions()

    def update_distributions(self):
        weight_sum = np.sum(self.weights)
        results = (1.0 - self.gamma) * (self.weights / weight_sum) + (self.gamma / self.weights.shape[0])
        self.probability_distribution = results

    def provide_arm(self):
        drawn_price = self.rng.choice(self.arm_prices, size=1, p=self.probability_distribution, replace=False)[0]
        price_idx = np.where(self.arm_prices == drawn_price)[0]
        return drawn_price, int(price_idx)

    def update_estimations(self, price_idx, reward):
        reward = float(reward)
        estimated_rewards = reward / self.probability_distribution[price_idx]
        exp_value = estimated_rewards * self.gamma / self.arm_prices.shape[0]
        self.weights[price_idx] = self.weights[price_idx] * math.exp(exp_value)
        self.update_distributions()


class GPUCBEstimator(BaseGPEstimator):
    """
    An object that performs GP-UCB1 on Bernoulli-like arms
    It simply provides the estimations and updates them
    Optimizing and playing are left to the caller

     GP-UCB implementation as described in "Gaussian Process
     Optimization in the Bandit Setting: No Regret and Experimental Design"
     by Srinivas et al.
     https://icml.cc/Conferences/2010/papers/422.pdf
    """

    def __init__(self, arms: np.ndarray, kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float):
        """
        :param arms: the available arms (here arms are the true values, not their indices)
        :param kernel: the kernel to be used by the GP
        :param alpha: the alpha to be used by the GP
        :param beta: the parameter used to compute the upper/lower bound by UCB
        """
        super().__init__(arms, kernel, alpha)
        self.beta = beta

    def provide_estimations(self, lower_bound=False):
        """
        Provides the estimation of the rewards
        Update the model at least once before calling this
        :param lower_bound: True if you want a lower bound, false otherwise
        :return: the estimations (np array)
        """

        # just a detail to make sure that sigma is at least a small epsilon (from the lab)
        sigmas = np.maximum(self.sigma_vector, 1e-2)
        if lower_bound:
            thetas = self.mu_vector - sigmas * np.sqrt(self.beta)
        else:
            thetas = self.mu_vector + sigmas * np.sqrt(self.beta)
        return thetas


class GPTSEstimator(BaseGPEstimator):
    """
    An object that performs GP-TS on Bernoulli-like arms
    It simply provides the estimations and updates them
    Optimizing and playing are left to the caller

    """

    def __init__(self, arms: np.ndarray, kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float,
                 rng: np.random.Generator):
        """
        :param arms: the available arms (here arms are the true values, not their indices)
        :param kernel: the kernel to be used by the GP
        :param alpha: the alpha to be used by the GP
        :param rng: the numpy random number generator to be used
        """
        super().__init__(arms, kernel, alpha)
        self.rng = rng

    def provide_estimations(self):
        """
        Provides the estimation of the rewards
        Works also if no data is available
        :return: the estimations (np array)
        """
        # just a detail to make sure that sigma is at least a small epsilon (from the lab)
        sigmas = np.maximum(self.sigma_vector, 1e-2)
        thetas = self.rng.normal(self.mu_vector, sigmas)
        return thetas
