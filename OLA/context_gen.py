import numpy as np
import sklearn

import environments as envs
from estimators import BaseGPEstimator
from learners import single_class_opt


def split_data(data: dict, feature: int):
    # Only the set of user profiles is changed, the rest of the data is kept the same

    data_f0 = dict()
    data_f0.update(data)
    data_f0["profiles"] = set(filter(lambda p: p[feature] == 0, data["profiles"]))

    data_f1 = dict()
    data_f1.update(data)
    data_f1["profiles"] = data["profiles"] - data_f0["profiles"]

    return data_f0, data_f1


def get_n_users(data: dict):
    n = 0
    for profile in data["profiles"]:
        n += np.sum(data["clicks"][profile])
    return n


class ContextGeneration:
    def __init__(self, environment: envs.MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                 kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, rng: np.random.Generator,
                 beta: float, bound_confidence: float):
        self.env = environment
        self.bids = bids
        self.prices = prices
        self.kernel = kernel
        self.alpha = alpha
        self.beta = beta  # parameter to compute UCB-like lower (or upper) bound
        self.rng = rng
        self.log_confidence = np.log(bound_confidence)

    def bound(self, value, set_size, upper=False):
        return value - np.sqrt(-self.log_confidence / 2 / set_size) * (-1 if upper else 1)

    def ucb_like_bound(self, value, sigmas, upper=False):
        return value - np.maximum(sigmas, 1e-2) * np.sqrt(self.beta) * (-1 if upper else 1)

    def evaluate_data(self, data: dict):
        convs = np.zeros(self.prices)
        clicks = np.zeros(self.prices)

        clicks_gp = BaseGPEstimator(self.bids, self.kernel, self.alpha)
        costs_gp = BaseGPEstimator(self.bids, self.kernel, self.alpha)

        for profile in data["profiles"]:
            for t, price in enumerate(data["prices"][profile]):
                convs[price] += data["conversions"][profile][t]
                clicks[price] += data["clicks"][profile][t]

                bid = data["bids"][profile][t]
                clicks_gp.update_model(bid, data["clicks"][profile][t])
                costs_gp.update_model(bid, data["costs"][profile][t])

        alphas_lower_bounds = self.bound(convs / clicks, clicks)
        clicks_lower_bounds = self.ucb_like_bound(clicks_gp.mu_vector, clicks_gp.sigma_vector)
        costs_upper_bounds = self.ucb_like_bound(costs_gp.mu_vector, costs_gp.sigma_vector, True)

        bid, price = single_class_opt(self.bids, self.prices, alphas_lower_bounds, clicks_lower_bounds,
                                      costs_upper_bounds)
        return price * alphas_lower_bounds[price] * clicks_lower_bounds[bid] - costs_upper_bounds[bid]

    def evaluate_feature(self, data: dict, feature: int):
        data_f0, data_f1 = split_data(data, feature)
        tot_users = get_n_users(data)
        p_f0 = self.bound(get_n_users(data_f0) / tot_users, tot_users)
        p_f1 = self.bound(get_n_users(data_f1) / tot_users, tot_users)
        return p_f0 * self.evaluate_data(data_f0) + p_f1 * self.evaluate_data(data_f1)

    def decide_split(self, data: dict, features: list):
        no_split_value = self.evaluate_data(data)
        split_values = [self.evaluate_feature(data, f) for f in features]

        if max(split_values) > no_split_value:
            return True, features[np.argmax(split_values)]
        else:
            return False, None

    def generate(self, data: dict, features: list):
        if sorted(list(features)) != [0, 1]:
            raise ValueError("This implementation only works with features [0, 1]")

        root_split_flag, root_f = self.decide_split(data, features)

        contexts = []
        if root_split_flag:
            features = [1 - root_f]
            data_left, data_right = split_data(data, root_f)

            left_split_flag, left_f = self.decide_split(data_left, features)
            if left_split_flag:
                data_ll, data_lr = split_data(data_left, left_f)
                contexts.append(list(data_ll["profiles"]) + list(data_lr["profiles"]))
            else:
                contexts.append(list(data_left["profiles"]))

            right_split_flag, right_f = self.decide_split(data_right, features)
            if right_split_flag:
                data_rl, data_rr = split_data(data_right, right_f)
                contexts.append(list(data_rl["profiles"]) + list(data_rr["profiles"]))
            else:
                contexts.append(list(data_right["profiles"]))
        else:
            contexts.append(list(data["profiles"]))

        mapping = {}
        for profile in data["profiles"]:
            context_idx = list(map(lambda c: profile in c, contexts)).index(True)
            mapping[profile] = context_idx

        return len(contexts), mapping
