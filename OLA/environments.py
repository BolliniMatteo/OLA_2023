import numpy as np
from typing import Callable

"""
Here I define the environments 
and the objects that keep track of the history and compute statistics
"""

class SingleClassEnvironment:

    def __init__(self, N, en, C, ec, A, rng):
        """
        Parameters
        ----------
        N : callable
            N:bid->E[number of clicks]
        en : callable
            en() returns a sample from a gaussian distribution
        C : callable
            C:bid->E[payment for clicks]
        ec : callable
            ec() returns a sample from a gaussian distribution
        A : dictionary
            A[p] = E[conversion rate at price p]
        rng : random generator
            a numpy random generator to be used (mostly for Bernoulli)

        Returns
        -------
        None.

        """
        self.N = N
        self.en = en
        self.C = C
        self.ec = ec
        self.A = A
        self.rng = rng

    def perform_day(self, x: float, p: float):
        """
        :param x: the bid selected for the day
        :param p: the price selected for the day
        :return: (n,q,c)
        n: int - the number of clicks
        q: int - the number of conversions
        c: float - the advertising costs
        """
        n = self.N(x) + self.en()
        samples = self.rng.Binomial(n=1, p=self.A[p], size=n)
        q = np.sum(samples)
        c = self.C(x) + self.ec()
        return n, q, c


class SingleClassEnvironmentHistory:
    """
    History of all the steps performed by an environment
    Observe that it is not stored inside the environment itself, but by the learner
    """

    def __init__(self, N, C, A, best_bid, best_price):
        """
        Parameters
        ----------
        N : callable
            N:bid->E[number of clicks] must be applicable to arrays
        C : callable
            C:bid->E[payment for clicks] must be applicable to arrays
        A : dictionary
            A[p] = E[conversion rate at price p]
        best_bid : float
            the bid (among the available ones) that maximizes the reward
        best_price : float
            the price (among the available ones) that maximizes the reward

        Returns
        -------
        None.

        """
        self.N = N
        self.C = C
        self.A = A
        self.x_best = best_bid
        self.p_best = best_price

        self.xs = []
        self.ps = []
        self.ns = []
        self.qs = []
        self.cs = []

    def add_step(self, x: float, p: float, n: int, q: int, c: float):
        """
        Memorizes a new step (i.e., day) that has been performed

        Parameters
        ----------
        x : float
            the chosen bid
        p : float
            the chosen price
        n : int
            the number of clicks achieved
        q : int
            the number of conversions achieved
        c : float
            the advertising costs incurred in

        Returns
        -------
        None.

        """
        self.xs.append(x)
        self.ps.append(p)
        self.ns.append(n)
        self.qs.append(q)
        self.cs.append(c)

    def average_stats(self):
        """
        These stats are computed with the expected rewards
        (without noise)

        Returns
        -------
        instantaneous_rewards : numpy array
            the instantaneous rewards
        instantaneous_regrets : numpy array
            the instantaneous regrets
        cumulative_rewards : numpy array
            the cumulative rewards
        cumulative_regrets : numpy array
            the cumulative regrets

        """
        ps = np.array(self.ps)
        xs = np.array(self.xs)

        # first get the conversions
        # idea from https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
        alphas = np.vectorize(self.A.get)(ps)

        instantaneous_rewards = alphas * ps * self.N(xs) - self.C(xs)

        best_reward = self.A[self.p_best] * self.p_best * self.N(self.x_best) - self.C(self.x_best)

        instantaneous_regrets = best_reward - instantaneous_rewards

        return instantaneous_rewards, instantaneous_regrets, np.cumsum(instantaneous_rewards), np.cumsum(
            instantaneous_regrets)

    def played_rounds(self):
        return len(self.ps)


class MultiClassEnvironment:
    """
    The complete multi-class environment
    (works also when the estimated classes are not the true ones)
    """

    def __init__(self, n_features: int, class_map: dict, user_prob_map: dict,
                 n: list, en: Callable, c: list, ec: Callable, a: list, rng: np.random.Generator):
        """
        :param n_features: the number of features
        :param class_map: a mapping user_type->class (tuple->int), classes are from 0 to len(n)=len(c)=len(a)
        :param user_prob_map: a mapping user_type->probability
        :param n: a list of functions for the number of clicks
        :param en: the noise for the number of clicks
        :param c: a list of functions for the advertising costs
        :param ec: the noise for the advertising costs
        :param a: a list of dictionaries for the conversion rates
        :param rng: a numpy random number generator that will be used by this object
        """
        self.n_features = n_features
        self.class_map = class_map
        self.user_prob_map = user_prob_map
        self.n = n
        self.en = en
        self.c = c
        self.ec = ec
        self.a = a
        self.rng = rng
        self.user_profiles = class_map.keys()

    def perform_day(self, bids: dict, prices: dict):
        """
        :param bids: a mapping user_profile->bid
        :param prices: a mapping user_profile->price
        :return: a mapping user_profile->(n,q,c)
        n: int - the number of clicks for that user profile
        q: int - the number of conversions for that user profile
        c: float - the advertising costs for that user profile
        """
        result = {}
        for user_prof in self.user_profiles:
            bid = bids[user_prof]
            price = prices[user_prof]
            user_class = self.class_map[user_prof]
            user_prob = self.user_prob_map[user_prof]
            n = (self.n[user_class](bid) + self.en()) * user_prob
            n = int(n)
            # I am pretty sure that Binomial exists in the standard generator that we use
            samples = self.rng.Binomial(n=1, p=self.a[user_class][price], size=n)
            q = np.sum(samples)
            c = (self.c[user_class](bid) + self.ec()) * user_prob
            result[user_prof] = (n, q, c)
        return result

    def classes_count(self):
        return len(self.n)


class MultiClassEnvironmentHistory:
    def __init__(self, environment: MultiClassEnvironment, best_bids: np.ndarray, best_prices: np.ndarray):
        """
        :param environment: the multi class environment of which the history will be recorded
        :param best_bids: the best bid for each class
        :param best_prices: the best price for each class
        """
        self.env = environment
        self.best_bids = best_bids
        self.best_prices = best_prices
        self.xs = {}
        self.ps = {}
        self.ns = {}
        self.qs = {}
        self.cs = {}
        for user_profile in environment.user_profiles:
            self.xs[user_profile] = []
            self.ps[user_profile] = []
            self.ns[user_profile] = []
            self.qs[user_profile] = []
            self.cs[user_profile] = []

    def add_step(self, x: dict, p: dict, step_results: dict):
        """
        Memorizes a new step (i.e., day) that has been performed
        :param x: mapping user_profile->bid
        :param p: mapping user_profile->price
        :param step_results: mapping user_profile-> (n,q,c) as in MultiClassEnvironment.perform_day
        :return:
        """
        for user_profile in self.env.user_profiles:
            self.xs[user_profile].append(x[user_profile])
            self.ps[user_profile].append(p[user_profile])
            n, q, c = step_results[user_profile]
            self.ns[user_profile].append(n)
            self.qs[user_profile].append(q)
            self.cs[user_profile].append(c)

    def average_stats(self):
        # TODO: compute the average stats over an episode with multiple classes
        pass
