import numpy as np
import sklearn
from abc import ABC, abstractmethod

import environments as envs
import estimators as est
from environments import SingleClassEnvironmentHistory
from environments import MultiClassEnvironmentHistory

"""
The actual learners for the various step and the optimization functions that they use
A learner has to:
- instantiate the history object, computing the clairvoyant solution 
- instantiate and use the estimators that it needs
- at each round, get the estimations, optimize, play and update the estimators
"""


def single_class_price_opt(prices: np.ndarray, estimated_alphas: np.ndarray):
    vs = estimated_alphas * prices
    best_p_ind = np.argmax(vs)
    best_p = prices[best_p_ind]
    return best_p, best_p_ind


def single_class_bid_opt(bids: np.ndarray, price: float, alpha: float,
                         estimated_clicks: np.ndarray, estimated_costs: np.ndarray):
    objs = price * alpha * estimated_clicks - estimated_costs
    best_bid_ind = np.argmax(objs)
    best_bid = bids[best_bid_ind]
    return best_bid, best_bid_ind


def single_class_opt(bids: np.ndarray, prices: np.ndarray,
                     estimated_alphas: np.ndarray, estimated_clicks: np.ndarray, estimated_costs: np.ndarray):
    """
    Given the possible bids and prices and the estimated environment parameters,
    this function returns the best solution.
    Obs: for some steps, you use the actual values for some parameters and not the estimations,
    but the optimizer is the same
    :param bids: the available bids
    :param prices: the available prices
    :param estimated_alphas: the estimated conversion rates, estimated_alphas[i]= alpha for price=prices[i]
    :param estimated_clicks: the estimated number of clicks, estimated_clicks[i]= number of clicks for bid=bids[i]
    :param estimated_costs: the estimated advertising costs, estimated_costs[i]= number of clicks for bid=bids[i]
    :return: (bid, bid_index, price, price_index) the optimal bid and price to be played given the estimations
    and relative indices in the arrays bids and prices
    """

    best_p, best_p_ind = single_class_price_opt(prices, estimated_alphas)
    best_bid, best_bid_ind = single_class_bid_opt(bids, best_p, estimated_alphas[best_p_ind], estimated_clicks,
                                                  estimated_costs)
    return best_bid, best_bid_ind, best_p, best_p_ind

# these functions use np to optimize over multiple classes
# you can also optimize each class independently without using np parallelization,
# and probably you won't notice the difference


def multi_class_price_opt(prices: np.ndarray, estimated_alphas: np.ndarray):
    """
    :param prices: an array with the available prices
    :param estimated_alphas: a matrix where row c contains the alpha values estimated for class c
    :return: (ps,ps_ind)
    ps: np.ndarray contains the optimal prices (one for class)
    ps_ind: np.ndarray contains the optimal prices indices (according to the parameter prices)
    """
    # vs[class i] = estimated_alpha[class i] * prices
    vs = estimated_alphas * prices
    best_ps_ind = np.argmax(vs, axis=0)
    best_ps = prices[best_ps_ind]
    return best_ps, best_ps_ind


def multi_class_bid_opt(bids: np.ndarray, prices: np.ndarray, alphas: np.ndarray,
                        estimated_clicks: np.ndarray, estimated_costs: np.ndarray):
    """
    :param bids: the available bids
    :param prices: the (estimated) optimal prices, one for estimated class
    :param alphas: the (estimated) conversion rates with the optimal prices, one for class
    :param estimated_clicks: a matrix where row c contains the estimated number of clicks estimated for class c,
    shape=(n_classes,n_bids)
    :param estimated_costs: a matrix where row c contains the estimated advertising costs estimated for class c,
    shape=(n_classes,n_bids)
    :return: (bids, bids_ind)
    """
    vs = prices * alphas
    objs = estimated_clicks * vs - estimated_costs
    best_bids_ind = np.argmax(objs, axis=0)
    best_bids = bids[best_bids_ind]
    return best_bids, best_bids_ind


def multi_class_opt(bids: np.ndarray, prices: np.ndarray,
                    estimated_alphas: np.ndarray, estimated_clicks: np.ndarray, estimated_costs: np.ndarray):
    best_ps, best_ps_ind = multi_class_price_opt(prices, estimated_alphas)
    best_bids, best_bids_ind = multi_class_bid_opt(bids, best_ps,
                                                   estimated_alphas[np.arange(estimated_alphas.shape[0]), best_ps_ind],
                                                   estimated_clicks, estimated_costs)
    return best_bids, best_bids_ind, best_ps, best_ps_ind


class SingleClassLearner(ABC):
    def __init__(self, environment: envs.SingleClassEnvironment, bids: np.ndarray, prices: np.ndarray):
        self.env = environment
        self.ps = prices
        self.xs = bids
        self._prepare_history()

    # history uses the clairvoyant algorithm to compute the regret
    def _prepare_history(self):
        alphas_est = np.array([self.env.A[p] for p in self.ps])
        n_est = self.env.N(self.xs)
        c_est = self.env.C(self.xs)
        x_best, _, p_best, _ = single_class_opt(self.ps, self.xs, alphas_est, n_est, c_est)
        self.history = SingleClassEnvironmentHistory(self.env.N, self.env.C, self.env.A, x_best, p_best)

    def play_and_save(self, bid, price):
        n, q, c = self.env.perform_day(bid, price)
        self.history.add_step(bid, price, n, q, c)
        return n, q, c

    @abstractmethod
    def play_round(self):
        pass


class Step1UCBLearner(SingleClassLearner, ABC):
    def __init__(self, environment: envs.SingleClassEnvironment, bids: np.ndarray, prices: np.ndarray):
        super().__init__(environment, bids, prices)
        self.estimator = est.BeUCB1Estimator(self.ps.shape[0])

    def play_round(self):
        n_est = self.env.N(self.xs)
        c_est = self.env.C(self.xs)
        if self.history.played_rounds() < self.ps.shape[0]:
            # we have not played each arm at least once
            # we could also use the +inf given by the estimator and optimize as usually...
            p_t = self.ps[self.history.played_rounds()]
            p_t_ind = self.history.played_rounds()
            # we have no data: we estimate 0.5 for the optimization
            alpha = 0.5
            x_t, x_t_ind = single_class_bid_opt(self.xs, p_t, alpha, n_est, c_est)
        else:
            alphas_est = self.estimator.provide_estimations(lower_bound=False)
            x_t, x_t_ind, p_t, p_t_ind = single_class_opt(self.ps, self.xs, alphas_est, n_est, c_est)
        n, q, c = self.play_and_save(x_t, p_t)
        # ignore the warning, the argmax over the whole array is a single int and not an array
        self.estimator.update_estimations(p_t_ind, q, n)


class Step1TSLearner(SingleClassLearner):
    def __init__(self, environment: envs.SingleClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                 rng: np.random.Generator):
        super().__init__(environment, bids, prices)
        self.estimator = est.BeTSEstimator(self.ps.shape[0], rng)

    def play_round(self):
        alphas_est = self.estimator.provide_estimations()
        n_est = self.env.N(self.xs)
        c_est = self.env.C(self.xs)
        x_t, x_t_ind, p_t, p_t_ind = single_class_opt(self.ps, self.xs, alphas_est, n_est, c_est)
        n, q, c = self.play_and_save(x_t, p_t)
        # ignore the warning, the argmax over the whole array is a single int and not an array
        self.estimator.update_estimations(p_t_ind, q, n)


class Step2UCBLearner(SingleClassLearner):
    def __init__(self, environment: envs.SingleClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                 kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float):
        super().__init__(environment, bids, prices)
        self.n_estimator = est.GPUCB1Estimator(bids, kernel, alpha, beta)
        self.c_estimator = est.GPUCB1Estimator(bids, kernel, alpha, beta)
        self.alphas = np.array([self.env.A[p] for p in prices])
        # in theory, I could compute here the best price
        # and then optimize just the bid,
        # but I will simply use the single_class_opt function
        # and let it re-compute the constant optimal price at each round

    def play_round(self):
        alphas_est = self.alphas
        n_est = self.n_estimator.provide_estimations(lower_bound=False)
        c_est = self.c_estimator.provide_estimations(lower_bound=True)
        x_t, x_t_ind, p_t, p_t_ind = single_class_opt(self.ps, self.xs, alphas_est, n_est, c_est)
        n, _, c = self.play_and_save(x_t, p_t)
        self.n_estimator.update_model(x_t, n)
        self.c_estimator.update_model(x_t, c)


class Step2TSLearner(SingleClassLearner):
    def __init__(self, environment: envs.SingleClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                 kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, rng: np.random.Generator):
        super().__init__(environment, bids, prices)
        self.n_estimator = est.GPTSEstimator(bids, kernel, alpha, rng)
        self.c_estimator = est.GPTSEstimator(bids, kernel, alpha, rng)
        self.alphas = np.array([self.env.A[p] for p in prices])
        # in theory, I could compute here the best price
        # and then optimize just the bid,
        # but I will simply use the single_class_opt function
        # and let it re-compute the constant optimal price at each round

    def play_round(self):
        alphas_est = self.alphas
        n_est = self.n_estimator.provide_estimations()
        c_est = self.c_estimator.provide_estimations()
        x_t, x_t_ind, p_t, p_t_ind = single_class_opt(self.ps, self.xs, alphas_est, n_est, c_est)
        n, _, c = self.play_and_save(x_t, p_t)
        self.n_estimator.update_model(x_t, n)
        self.c_estimator.update_model(x_t, c)


class Step3UCBLearner(SingleClassLearner):
    def __init__(self, environment: envs.SingleClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                 kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float):
        super().__init__(environment, bids, prices)
        self.a_estimator = est.BeUCB1Estimator(prices.shape[0])
        self.n_estimator = est.GPUCB1Estimator(bids, kernel, alpha, beta)
        self.c_estimator = est.GPUCB1Estimator(bids, kernel, alpha, beta)

    def play_round(self):
        if self.history.played_rounds() < self.ps.shape[0]:
            # we have not played each arm (price) at least once, and for price there are no GP
            # obs: we could also use the default +inf from the estimator,
            # but probably keeping the special case and assigning alpha=0.5 is better
            p_t = self.ps[self.history.played_rounds()]
            p_t_ind = self.history.played_rounds()
            # we have no data: we estimate 0.5 for the optimization
            alpha = 0.5
        else:
            alphas_est = self.a_estimator.provide_estimations(lower_bound=False)
            p_t, p_t_ind = single_class_price_opt(self.ps, alphas_est)
            alpha = alphas_est[p_t_ind]

        n_est = self.n_estimator.provide_estimations(lower_bound=False)
        c_est = self.c_estimator.provide_estimations(lower_bound=True)
        x_t, x_t_ind = single_class_bid_opt(self.xs, p_t, alpha, n_est, c_est)
        n, q, c = self.play_and_save(x_t, p_t)
        # ignore the warning: the argmax over the whole array returns an int, not an array
        self.a_estimator.update_estimations(p_t_ind, q, n)
        self.n_estimator.update_model(x_t, n)
        self.c_estimator.update_model(x_t, c)


class Step3TSLearner(SingleClassLearner):
    def __init__(self, environment: envs.SingleClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                 kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, rng: np.random.Generator):
        super().__init__(environment, bids, prices)
        self.a_estimator = est.BeTSEstimator(prices.shape[0], rng)
        self.n_estimator = est.GPTSEstimator(bids, kernel, alpha, rng)
        self.c_estimator = est.GPTSEstimator(bids, kernel, alpha, rng)

    def play_round(self):
        alphas_est = self.a_estimator.provide_estimations()
        n_est = self.n_estimator.provide_estimations()
        c_est = self.c_estimator.provide_estimations()
        x_t, x_t_ind, p_t, p_t_ind = single_class_opt(self.ps, self.xs, alphas_est, n_est, c_est)
        n, q, c = self.play_and_save(x_t, p_t)
        # ignore the warning: the argmax over the whole array returns an int, not an array
        self.a_estimator.update_estimations(p_t_ind, q, n)
        self.n_estimator.update_model(x_t, n)
        self.c_estimator.update_model(x_t, c)


class MultiClassLearner(ABC):
    def __init__(self, environment: envs.MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray):
        self.env = environment
        self.ps = prices
        self.xs = bids
        self._prepare_history()

    # history uses the clairvoyant algorithm to compute the regret
    def _prepare_history(self):
        alphas = np.array([[self.env.a[c][p] for p in self.ps] for c in range(self.env.classes_count())])
        ns = np.array([self.env.n[c](self.xs) for c in self.env.classes_count()])
        cs = np.array([self.env.c[c](self.xs) for c in self.env.classes_count()])
        xs_best, _, ps_best, _ = multi_class_opt(self.ps, self.xs, alphas, ns, cs)
        self.history = MultiClassEnvironmentHistory(self.env, xs_best, ps_best)

    def play_and_save_raw(self, bids, prices):
        """
        :param bids: a mapping user_profile->bid
        :param prices: a mapping user_profile->price
        :return: a mapping user_profile->(n,q,c)
        n: int - the number of clicks for that user profile
        q: int - the number of conversions for that user profile
        c: float - the advertising costs for that user profile
        """
        results = self.env.perform_day(bids, prices)
        self.history.add_step(bids, prices, results)
        return results

    def play_and_save(self, class_map: dict, bids: np.ndarray, prices: np.ndarray):
        """
        A "better" play_and_save that takes the estimated classes and the optimized inputs for them,
        produces the "raw" inputs for the environment and plays.
        Results are divided by user type and not estimated classes,
        so that your estimation of the classes can be refined
        :param class_map: a mapping user_type->estimated_class (tuple->int),
        estimated classes are from 0 to bids.shape[0]=prices.shape[0]
        :param bids: an array with the bid for each estimated class
        :param prices: an array with the price for eacc estimated class
        :return: a mapping user_profile->(n,q,c)
        n: int - the number of clicks for that user profile
        q: int - the number of conversions for that user profile
        c: float - the advertising costs for that user profile
        """
        bids_dict = {user_profile: bids[class_map[user_profile]] for user_profile in self.env.user_profiles}
        prices_dict = {user_profile: prices[class_map[user_profile]] for user_profile in self.env.user_profiles}
        return self.play_and_save_raw(bids_dict, prices_dict)

    def step_results_to_estimated_classes(self, step_results: dict, class_map: dict):
        """
        Converts the results of play_and_save to the estimated classes.
        Useful during rounds where you don't change the current estimation of the classes
        :param step_results: results from play_and_save
        :param class_map: class_map: a mapping user_type->estimated_class (tuple->int)
        :return: a list [(n,q,c) for class in range(n_estimated_classes)]
        n: int - the number of clicks for that class
        q: int - the number of conversions for that class
        c: float - the advertising costs for that class
        """
        n_classes = max(class_map.values()) + 1
        results = [(0, 0, 0) for _ in range(n_classes)]
        for user_profile in self.env.user_profiles:
            old_n, old_q, old_c = results[class_map[user_profile]]
            n, q, c = step_results[user_profile]
            n = old_n + n
            q = old_q + q
            c = old_c + c
            results[class_map[user_profile]] = (n, q, c)
        return results

    @abstractmethod
    def play_round(self):
        pass


class Step4TSClassesKnownLearner(MultiClassLearner):
    def __init__(self, environment: envs.MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                 kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, rng: np.random.Generator):
        super().__init__(environment, bids, prices)
        self.a_estimators = [est.BeTSEstimator(prices.shape[0], rng) for _ in range(self.env.classes_count())]
        self.n_estimators = [est.GPTSEstimator(bids, kernel, alpha, rng) for _ in range(self.env.classes_count())]
        self.c_estimators = [est.GPTSEstimator(bids, kernel, alpha, rng) for _ in range(self.env.classes_count())]

    def play_round(self):
        alphas_est = np.array([self.a_estimators[c].provide_estimations() for c in range(self.env.classes_count())])
        n_est = np.array([self.n_estimators[c].provide_estimations() for c in range(self.env.classes_count())])
        c_est = np.array([self.c_estimators[c].provide_estimations() for c in range(self.env.classes_count())])
        xs_t, xs_t_ind, ps_t, ps_t_ind = multi_class_opt(self.ps, self.xs, alphas_est, n_est, c_est)
        results = self.play_and_save(self.env.class_map, xs_t, ps_t)
        results = self.step_results_to_estimated_classes(results, self.env.class_map)
        for c in range(len(results)):
            n, q, c = results[c]
            self.a_estimators[c].update_estimations(ps_t_ind[c], q, n)
            self.n_estimators[c].update_model(xs_t[c], n)
            self.c_estimators[c].update_model(xs_t[c], c)
