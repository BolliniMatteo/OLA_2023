from Learner import *
import numpy as np

"""
Simple UCB1 learner, implements the vanilla version of the algorithm
"""


class UCB1Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)

    def pull_arm(self):
        # Play each arm at least once before applying algorithm
        if self.t < self.n_arms:
            return self.t

        # Compute reward mean for each arm
        mean_vector = [np.mean(arm_rewards, axis=0) for arm_rewards in self.rewards_per_arm]

        # Compute number of pulls for each arm
        length_vector = [len(self.rewards_per_arm[i]) for i in range(len(self.rewards_per_arm))]

        # Vector of confidence terms (one per arm)
        confidence_terms = np.sqrt((2 * np.log(self.t)) / np.array(length_vector))

        # Select most promising arm
        idx = np.argmax(np.array(mean_vector) + confidence_terms)
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
