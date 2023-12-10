import numpy as np
import sklearn

import environments as envs
from estimators import BaseGPEstimator
from OLA.optimization_utilities import single_class_opt
from environments import SingleClassHistory, MultiClassEnvironmentHistory
from abc import ABC, abstractmethod


class ContextData:
    def __init__(self, profiles: set = None, bids: dict = None, prices: dict = None, clicks: dict = None,
                 conversions: dict = None, costs: dict = None, prod_cost: float = None):
        self.profiles = profiles
        self.bids = bids
        self.prices = prices
        self.clicks = clicks
        self.conversions = conversions
        self.costs = costs
        self.prod_cost = prod_cost


def get_data_from_history(history: MultiClassEnvironmentHistory):
    # return {
    #     "profiles": set(self.env.user_profiles),
    #     "bids": self.xs,
    #     "prices": self.ps,
    #     "clicks": self.ns,
    #     "conversions": self.qs,
    #     "costs": self.cs
    # }
    return ContextData(set(history.env.user_profiles), history.xs, history.ps, history.ns,
                       history.qs, history.cs, history.env.prod_cost)

def copy_data(data: ContextData):
    return ContextData(data.profiles, data.bids, data.prices,
                       data.clicks, data.conversions, data.costs, data.prod_cost)


def split_data(data: ContextData, feature: int):
    # Only the set of user profiles is changed, the rest of the data is kept the same
    # but will be filtered on usage because some profiles are missing
    data_f0 = copy_data(data)
    data_f0.profiles = set(filter(lambda p: p[feature] == 0, data.profiles))
    data_f1 = copy_data(data)
    data_f1.profiles = data.profiles - data_f0.profiles
    return data_f0, data_f1


def get_n_users(data: ContextData):
    n = 0
    for profile in data.profiles:
        n += np.sum(data.clicks[profile])
    return n


class ContextGenerator:
    def __init__(self, environment: envs.MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                 kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, rng: np.random.Generator, beta: float,
                 bound_confidence: float):
        self.env = environment
        self.bids = bids
        self.prices = prices
        self.kernel = kernel
        self.alpha = alpha
        self.beta = beta  # parameter to compute UCB-like lower (or upper) bound
        self.rng = rng
        self.log_confidence = np.log(bound_confidence)

    def bound(self, value, set_size, upper=False):
        return value - np.sqrt(-self.log_confidence / (2 * set_size)) * (-1 if upper else 1)

    def ucb_like_bound(self, value, sigmas, upper=False):
        return value - np.maximum(sigmas, 1e-2) * np.sqrt(self.beta) * (-1 if upper else 1)

    def evaluate_data(self, data: ContextData,
                      click_estimators: dict, cost_estimators: dict):
        # we are trying to evaluate the data if we played as if it was a single class,
        # but we didn't: what if two user profiles had tow different bid the same day?
        # we need to estimate the number of clicks if we play the same bid with both profiles
        # an idea is to estimate independently (with lower bound) the curve of each user profile
        # then we do sum them
        # Observe that the sum is not weighted: if a user profile has less chance to happen,
        # this is reflected to its number of clicks
        # Finally observe that with the conversion rate we don't have such a problem,
        # just sum everything and then compute the mean

        convs = np.zeros(self.prices.shape)
        clicks_for_price = np.zeros(self.prices.shape)
        # these are the clicks for price, used to estimate alpha
        # the bound on alpha will be the H. one
        for profile in data.profiles:
            for t, price in enumerate(data.prices[profile]):
                # trick: find the index of price in the array prices
                price_index = np.argmax(self.prices == price)
                convs[price_index] += data.conversions[profile][t]
                clicks_for_price[price_index] += data.clicks[profile][t]
        alphas_lower_bounds = self.bound(convs / clicks_for_price, clicks_for_price)

        # these are the curves estimated by the gps
        clicks_lower_bounds = np.zeros(self.bids.shape)
        costs_upper_bounds = np.zeros(self.bids.shape)
        for profile in data.profiles:
            clicks_lower_bounds += self.ucb_like_bound(click_estimators[profile].mu_vector,
                                                       click_estimators[profile].sigma_vector)
            costs_upper_bounds += self.ucb_like_bound(cost_estimators[profile].mu_vector,
                                                      cost_estimators[profile].sigma_vector, True)

        bid, bid_index, price, price_index = single_class_opt(self.bids, self.prices, alphas_lower_bounds,
                                                              clicks_lower_bounds, costs_upper_bounds, data.prod_cost)
        return SingleClassHistory.reward(price, alphas_lower_bounds[price_index],
                                         clicks_lower_bounds[bid_index], costs_upper_bounds[bid_index], data.prod_cost)

    def evaluate_feature(self, data: ContextData, feature: int, click_estimators: dict, cost_estimators: dict):
        data_f0, data_f1 = split_data(data, feature)
        tot_users = get_n_users(data)
        p_f0 = self.bound(get_n_users(data_f0) / tot_users, tot_users)
        p_f1 = self.bound(get_n_users(data_f1) / tot_users, tot_users)
        return (p_f0 * self.evaluate_data(data_f0, click_estimators, cost_estimators)
                + p_f1 * self.evaluate_data(data_f1, click_estimators, cost_estimators))

    def decide_split(self, data: ContextData, features: list, click_estimators: dict, cost_estimators: dict):
        no_split_value = self.evaluate_data(data, click_estimators, cost_estimators)
        split_values = [self.evaluate_feature(data, f, click_estimators, cost_estimators) for f in features]

        if max(split_values) > no_split_value:
            return True, features[np.argmax(split_values)]
        else:
            return False, None

    # def compute_estimators(self, data: ContextData):
    #     click_estimators = dict()
    #     cost_estimators = dict()
    #     for profile in data.profiles:
    #         click_estimators[profile] = BaseGPEstimator(self.bids, self.kernel, self.alpha)
    #         click_estimators[profile].update_model(data.bids[profile], data.clicks[profile])
    #         cost_estimators[profile] = BaseGPEstimator(self.bids, self.kernel, self.alpha)
    #         cost_estimators[profile].update_model(data.bids[profile], data.costs[profile])
    #     return click_estimators, cost_estimators

    def generate(self, data: ContextData, features: list, click_estimators: dict, cost_estimators: dict):
        if sorted(list(features)) != [0, 1]:
            raise ValueError("This implementation only works with features {0, 1}")

        # we can generate these once for the whole dataset,
        # then at each evaluation we use just those we need

        root_split_flag, root_f = self.decide_split(data, features, click_estimators, cost_estimators)

        contexts = []
        if root_split_flag:
            features = [1 - root_f]
            data_left, data_right = split_data(data, root_f)

            left_split_flag, left_f = self.decide_split(data_left, features, click_estimators, cost_estimators)
            if left_split_flag:
                data_ll, data_lr = split_data(data_left, left_f)
                contexts.append(list(data_ll.profiles) + list(data_lr.profiles))
            else:
                contexts.append(list(data_left.profiles))

            right_split_flag, right_f = self.decide_split(data_right, features, click_estimators, cost_estimators)
            if right_split_flag:
                data_rl, data_rr = split_data(data_right, right_f)
                contexts.append(list(data_rl.profiles) + list(data_rr.profiles))
            else:
                contexts.append(list(data_right.profiles))
        else:
            contexts.append(list(data.profiles))

        mapping = {}
        for profile in data.profiles:
            context_idx = list(map(lambda c: profile in c, contexts)).index(True)
            mapping[profile] = context_idx

        return len(contexts), mapping
