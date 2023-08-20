import numpy as np
from Environment import *

"""
Environment for single class case and pricing
"""


class SingleClassPricingEnvironment(Environment):
    def __init__(self, n_arms, probabilities=None):
        super().__init__()
        self.n_arms = n_arms
        self.probabilities = probabilities

    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward

