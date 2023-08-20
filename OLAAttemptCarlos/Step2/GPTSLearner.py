from Learner import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

"""
GP-TS implementation as shown during the course
"""


class GPTSLearner(Learner):
    def __init__(self, n_arms, arms):
        super().__init__(n_arms)
        self.arms = arms
        self.mu_vector = np.zeros(self.n_arms)
        self.sigma_vector = np.ones(self.n_arms) * 10
        self.pulled_arms = []
        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2, normalize_y=False)

    def pull_arm(self):
        sampled_values = np.random.normal(self.mu_vector, self.sigma_vector)
        return np.argmax(sampled_values)

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x, y)
        self.mu_vector, self.sigma_vector = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigma_vector = np.maximum(self.sigma_vector, 1e-2)

    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()
