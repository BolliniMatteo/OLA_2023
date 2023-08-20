from Learner import *
import numpy as np

"""
    Simple TS learner, implements its vanilla version
"""


class TSLearner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)

        # Initialize Beta distributions for each arm
        self.beta_parameters = np.ones((n_arms, 2))

    def pull_arm(self):
        # Sample Beta distributions for each arm and select the highest one
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]))
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)

        # Update beta parameters
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - reward
