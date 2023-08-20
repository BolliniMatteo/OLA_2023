import numpy as np

"""
Base learner class that will be extended to make room for more 
elaborate learners
"""


class Learner:

    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = [[] for i in range(n_arms)]
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward):
        """
        Updates observations (rewards obtained) for
        each arm
        :param pulled_arm: arm pulled by the learner
        :param reward: Reward given by the environment
        """
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
