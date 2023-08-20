from Learner import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

"""
 GP-UCB implementation as described in "Gaussian Process 
 Optimization in the Bandit Setting: No Regret and Experimental Design" 
 by Srinivas et al.
 https://icml.cc/Conferences/2010/papers/422.pdf
"""


class GPUCBLearner(Learner):
    def __init__(self, n_arms, arms):
        super().__init__(n_arms)
        self.arms = arms
        self.mu_vector = np.zeros(self.n_arms)
        # We choose a standard deviation of 3
        self.sigma_vector = np.ones(self.n_arms) * 3
        self.pulled_arms = []
        # I made some dumb calculations trying random numbers
        # according to the optimal beta formula shown in the paper, and it's around
        # 110 in our case (maybe)
        self.beta = 110

        alpha = 10.0
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha**2, normalize_y=False)

    def pull_arm(self):
        sampled_values = self.mu_vector + self.sigma_vector * np.sqrt(self.beta)
        return np.argmax(sampled_values)

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x, y)
        self.mu_vector, self.sigma_vector = self.gp.predict(
            np.atleast_2d(self.arms).T, return_std=True
        )

    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

