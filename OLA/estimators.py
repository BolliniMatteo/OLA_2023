import numpy as np
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor

"""
I try to keep things modular
Estimator simply estimate things without optimizing
They receive data and produce estimations, but do not play

I try to keep them independent, so they keep their state
and do not rely on the environment history
"""


class BeUCB1Estimator:
    """
    An object that performs UCB1 on Bernoulli-like arms
    It simply provides the estimations (mean + UCB bound) and updates them
    Optimizing and playing are left to the client

    Arms are referred as indices from 0 to "number of arms - 1"
    You may want to use a dictionary, list or other sort of mappings with the true value
    """

    def __init__(self, na: int):
        """
        :param na: the number of arms
        """
        self.play_counts = np.zeros(na, dtype=int)
        self.means = np.zeros(na, dtype=float)

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
            thetas[non_zero_mask] = thetas[non_zero_mask] - np.sqrt((2*np.log(t))/self.play_counts)
        else:
            thetas[zero_mask] = float('+inf')
            thetas[non_zero_mask] = thetas[non_zero_mask] + np.sqrt((2*np.log(t))/self.play_counts)
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

    def update_model(self, played_arm: int, reward: int | float):
        """
        Updates the internal attributes for the estimations
        :param played_arm: the arm that has been played in the last round
        :param reward: the reward obtained in the last round
        :return: None
        """
        self.played_arms.append(played_arm)
        self.rewards.append(reward)

        x = np.array(self.played_arms)
        X = np.reshape(x, (x.shape[0], 1))
        y = np.array(self.rewards)
        self.gp.fit(X, y)
        self.mu_vector, self.sigma_vector = self.gp.predict(np.atleast_2d(self.arms).T,
                                                            return_std=True)


class GPUCB1Estimator(BaseGPEstimator):
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

    def update_model(self, played_arm: int, reward: int | float):
        super().update_model(played_arm, reward)
