import numpy as np
from Environment import *

"""
Environment for single class case estimating budget and number of clicks
"""


class SingleClassBiddingEnvironment(Environment):
    def __init__(self, bids, sigma, fun):
        super().__init__()
        self.bids = bids
        self.means = fun(bids)
        self.sigmas = np.ones(len(bids)) * sigma

    def round(self, pulled_arm):
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])
