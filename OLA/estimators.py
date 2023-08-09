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
        Do not call this method till you haven't player each arm at least one
        :param lower_bound: True if you want a lower bound, false otherwise
        :return: the estimations (np array)
        """
        t = np.sum(self.play_counts)
        if lower_bound:
            thetas = self.means - np.sqrt((2*np.log(t))/self.means)
        else:
            thetas = self.means + np.sqrt((2*np.log(t))/self.means)
        return thetas

    def update_estimations(self, played_arm: int, positive_rewards: int, total_rewards: int):
        """
        Updates the internal attributes for the estimations
        :param played_arm: the arm that has been played in the last round
        :param positive_rewards: the number of positive rewards (r=1) collected in the last round
        :param total_rewards: the total rewards collected in the last round
        :return: None
        """
        self.means[played_arm] = self.means[played_arm]*self.play_counts[played_arm] + positive_rewards
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
    it just performs the updates to the GP
    """
    def __init__(self, arms: np.ndarray, kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float):
        """
        :param arms: the available arms (here arms are the true values, not their indices)
        :param kernel: the kernel to be used by the GP
        :param alpha: the alpha to be used by the GP
        """
        self.arms = arms
        # we are not estimating Bernoulli... is normalize_y=True right in this case?
        # I think yes: in the lab lecture it's not estimating Bernoulli variables
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=9)
        # this data could be found also in the environment history, but I prefer to keep this class independent
        self.played_arms = []
        self.rewards = []

    def update_model(self, played_arm: int, reward: int):
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


class GPUCB1Estimator(BaseGPEstimator):
    """
    An object that performs GP-UCB1 on Bernoulli-like arms
    It simply provides the estimations and updates them
    Optimizing and playing are left to the caller

    I'm not sure of how GP-UCB works.
    I assume you just use the mean+-variance interval that the CP gives you,
    but probably this is wrong and the interval should use also some sort of UCB-like term different from the variance
    (we need it to increase if the arm is not pulled for some time)

    """

    def __init__(self, arms: np.ndarray, kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float):
        """
        :param arms: the available arms (here arms are the true values, not their indices)
        :param kernel: the kernel to be used by the GP
        :param alpha: the alpha to be used by the GP
        """
        super().__init__(arms, kernel, alpha)

    def provide_estimations(self, lower_bound=False):
        """
        Provides the estimation of the rewards
        Update the model at least once before calling this
        :param lower_bound: True if you want a lower bound, false otherwise
        :return: the estimations (np array)
        """
        means, sigmas = self.gp.predict(self.arms.reshape((self.arms.shape[0], 1)), return_std=True)
        # just a detail to make sure that sigma is at least a small epsilon
        sigmas = np.maximum(sigmas, 1e-2)
        if lower_bound:
            thetas = means - sigmas
        else:
            thetas = means + sigmas
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
        self.updated_at_least_once = False

    def provide_estimations(self):
        """
        Provides the estimation of the rewards
        Works also if no data is available
        :return: the estimations (np array)
        """
        # this is the way the professor implements it in the videos
        # just notice that I don't store means and sigmas as attributes,
        # but I keep the flag "updated_at_least_once" for the first estimation
        if self.updated_at_least_once:
            means, sigmas = self.gp.predict(self.arms.reshape((self.arms.shape[0], 1)), return_std=True)
            # just a detail to make sure that sigma is at least a small epsilon
            sigmas = np.maximum(sigmas, 1e-2)
        else:
            means, sigmas = np.zeros(self.arms.shape), np.ones(self.arms.shape)*10
        thetas = self.rng.normal(means, sigmas)
        return thetas

    def update_model(self, played_arm: int, reward: int):
        super().update_model(played_arm, reward)
        self.updated_at_least_once = True
