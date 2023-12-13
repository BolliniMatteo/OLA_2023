from abc import ABC, abstractmethod

import numpy as np
import sklearn
from typing import Union

from OLA import environments as envs, estimators as est
from OLA.context_gen import ContextGenerator, StatefulContextGenerator, get_data_from_history
from OLA.environments import MultiClassEnvironmentHistory
from OLA.optimization_utilities import multi_class_opt, single_class_price_opt, multi_class_bid_opt


class MultiClassLearner(ABC):
    def __init__(self, environment: envs.MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray):
        self.env = environment
        self.ps = prices
        self.xs = bids
        self.history = MultiClassEnvironmentHistory(self.env)

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


class Step4TSContextGenLearner(MultiClassLearner):
    def __init__(self, environment: envs.MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                 kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, rng: np.random.Generator,
                 context_gen: ContextGenerator, burn_in: int):
        super().__init__(environment, bids, prices)
        self.kernel = kernel
        self.alpha = alpha
        self.rng = rng
        self.context_gen = context_gen
        self.burn_in = burn_in
        self.class_map = {p: 0 for p in self.env.user_profiles}
        n_classes = len(set(self.class_map.values()))
        self.a_estimators = [est.BeTSEstimator(self.ps.shape[0], self.rng) for _ in range(n_classes)]
        self.n_estimators = {profile: est.GPTSEstimator(bids, kernel, alpha, rng) for profile in self.env.user_profiles}
        self.c_estimators = {profile: est.GPTSEstimator(bids, kernel, alpha, rng) for profile in self.env.user_profiles}
        # the Be estimators is one for class.
        # the GPs instead are one for user profile.
        # If I played different bids at a certain round, when the user profiles are merged in a single context
        # I don't have a single sample for that round
        # So I use one estimator for profile, then I sum the estimations for profiles of the same context

    def play_round(self):
        played_rounds = self.history.played_rounds()
        if played_rounds > self.burn_in and played_rounds % 14 == 13:
            self._update_context()

        alphas_est, n_est, c_est = self._get_estimations()

        xs_t, xs_t_ind, ps_t, ps_t_ind = multi_class_opt(self.xs, self.ps, alphas_est, n_est, c_est, self.env.prod_cost)
        results = self.play_and_save(self.class_map, xs_t, ps_t)
        self._update_estimations(xs_t, ps_t_ind, results)

    def _update_context(self):
        data = get_data_from_history(self.history)
        n_classes, self.class_map = self.context_gen.generate(data, list(range(self.env.n_features)),
                                                              self.n_estimators, self.c_estimators)
        self.a_estimators = [est.BeTSEstimator(self.ps.shape[0], self.rng) for _ in range(n_classes)]
        # Initialize the Be estimator
        for t in range(self.history.played_rounds()):
            for profile in self.env.user_profiles:
                cl = self.class_map[profile]
                self.a_estimators[cl].update_estimations(
                    list(self.ps).index(data.prices[profile][t]),
                    data.conversions[profile][t], data.clicks[profile][t]
                )

    def _get_estimations(self):
        n_classes = len(set(self.class_map.values()))
        alphas_est = np.array([self.a_estimators[cl].provide_estimations() for cl in range(n_classes)])

        n_est = np.zeros((n_classes, self.xs.shape[0]))
        c_est = np.zeros((n_classes, self.xs.shape[0]))
        for profile in self.env.user_profiles:
            n_est += self.n_estimators[profile].provide_estimations()
            c_est += self.c_estimators[profile].provide_estimations()
        return alphas_est, n_est, c_est

    def _update_estimations(self, played_bids: np.ndarray, played_price_indices: np.ndarray, results: dict):
        for profile in self.env.user_profiles:
            n, q, c = results[profile]
            cl = self.class_map[profile]
            self.a_estimators[cl].update_estimations(
                played_price_indices[cl], q, n)
            self.n_estimators[profile].update_model(played_bids[cl], n)
            self.c_estimators[profile].update_model(played_bids[cl], c)


class Step4TSFixedClassesLearner(MultiClassLearner):

    def __init__(self, environment: envs.MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                 kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, rng: np.random.Generator,
                 class_map: dict):
        super().__init__(environment, bids, prices)
        self.rng = rng
        self.class_map = class_map
        self.n_classes = len(set(self.class_map.values()))
        self.a_estimators = [est.BeTSEstimator(self.ps.shape[0], self.rng) for _ in range(self.n_classes)]
        self.n_estimators = [est.GPTSEstimator(bids, kernel, alpha, rng) for _ in range(self.n_classes)]
        self.c_estimators = [est.GPTSEstimator(bids, kernel, alpha, rng) for _ in range(self.n_classes)]

    def play_round(self):
        alphas_est = np.array([self.a_estimators[cl].provide_estimations() for cl in range(self.n_classes)])
        n_est = np.array([self.n_estimators[cl].provide_estimations() for cl in range(self.n_classes)])
        c_est = np.array([self.c_estimators[cl].provide_estimations() for cl in range(self.n_classes)])

        xs_t, xs_t_ind, ps_t, ps_t_ind = multi_class_opt(self.xs, self.ps, alphas_est, n_est, c_est, self.env.prod_cost)
        raw_results = self.play_and_save(self.class_map, xs_t, ps_t)
        results = self.step_results_to_estimated_classes(raw_results, self.class_map)
        for cl in range(self.n_classes):
            n, q, c = results[cl]
            self.a_estimators[cl].update_estimations(ps_t_ind[cl], q, n)
            self.n_estimators[cl].update_model(xs_t[cl], n)
            self.c_estimators[cl].update_model(xs_t[cl], c)


class Step4TSRealClassesLearner(Step4TSFixedClassesLearner):
    def __init__(self, environment: envs.MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                 kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, rng: np.random.Generator):
        super().__init__(environment, bids, prices, kernel, alpha, rng, environment.class_map)


class Step4TSOneClassLearner(Step4TSFixedClassesLearner):
    def __init__(self, environment: envs.MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                 kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, rng: np.random.Generator):
        super().__init__(environment, bids, prices, kernel, alpha, rng, {p: 0 for p in environment.user_profiles})


class Step4UCBContextGenLearner(MultiClassLearner):
    """
    A multi-class learner that uses a context gen to determine the contex each week
    after an initial "burn in" period.
    UCB estimators are used.
    """

    def __init__(self, environment: envs.MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                 kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float,
                 context_gen: ContextGenerator, burn_in: int, ucb_constant=1):
        super().__init__(environment, bids, prices)
        self.kernel = kernel
        self.alpha = alpha
        self.beta = beta
        self.context_gen = context_gen
        self.burn_in = burn_in
        self.ucb_constant = ucb_constant
        self.class_map = {p: 0 for p in self.env.user_profiles}
        n_classes = len(set(self.class_map.values()))
        self.a_estimators = [est.BeUCB1Estimator(prices.shape[0], ucb_constant) for _ in range(n_classes)]
        self.n_estimators = {profile: est.GPUCBEstimator(bids, kernel, alpha, beta) for profile in
                             self.env.user_profiles}
        self.c_estimators = {profile: est.GPUCBEstimator(bids, kernel, alpha, beta) for profile in
                             self.env.user_profiles}

    def play_round(self):
        played_rounds = self.history.played_rounds()
        if played_rounds >= self.burn_in and played_rounds % 14 == 13:
            self._update_context()

        n_classes = len(set(self.class_map.values()))

        alphas = np.zeros(n_classes)
        ps_t = np.zeros(n_classes)
        ps_t_ind = np.zeros(n_classes, dtype=int)

        for cl in range(n_classes):
            if self.a_estimators[cl].get_non_pulled_arms().shape[0] == 0:
                alphas_est = self.a_estimators[cl].provide_estimations()
                ps_t[cl], ps_t_ind[cl] = single_class_price_opt(self.ps, alphas_est, self.env.prod_cost)
                alphas[cl] = alphas_est[ps_t_ind[cl]]
            else:
                ps_t_ind[cl] = self.a_estimators[cl].get_non_pulled_arms()[0]
                ps_t[cl] = self.ps[ps_t_ind[cl]]
                alphas[cl] = 0.3

        n_est, c_est = self._get_estimations()
        xs_t, xs_t_ind = multi_class_bid_opt(self.xs, ps_t, alphas, n_est, c_est, self.env.prod_cost)
        results = self.play_and_save(self.class_map, xs_t, ps_t)
        self._update_estimations(xs_t, ps_t_ind, results)

    def _update_context(self):
        data = get_data_from_history(self.history)
        n_classes, self.class_map = self.context_gen.generate(data, list(range(self.env.n_features)),
                                                              self.n_estimators, self.c_estimators)
        self.a_estimators = [est.BeUCB1Estimator(self.ps.shape[0], self.ucb_constant) for _ in range(n_classes)]
        # Initialize the Be estimators
        for t in range(self.history.played_rounds()):
            for profile in self.env.user_profiles:
                cl = self.class_map[profile]
                self.a_estimators[cl].update_estimations(
                    list(self.ps).index(data.prices[profile][t]),
                    data.conversions[profile][t], data.clicks[profile][t]
                )

    def _get_estimations(self):
        n_classes = len(set(self.class_map.values()))
        n_est = np.zeros((n_classes, self.xs.shape[0]))
        c_est = np.zeros((n_classes, self.xs.shape[0]))
        for profile in self.env.user_profiles:
            n_est += self.n_estimators[profile].provide_estimations(lower_bound=False)
            c_est += self.c_estimators[profile].provide_estimations(lower_bound=True)
        return n_est, c_est

    def _update_estimations(self, played_bids: np.ndarray, played_price_indices: np.ndarray, results: dict):
        for profile in self.env.user_profiles:
            n, q, c = results[profile]
            cl = self.class_map[profile]
            self.a_estimators[cl].update_estimations(
                played_price_indices[cl], q, n)
            self.n_estimators[profile].update_model(played_bids[cl], n)
            self.c_estimators[profile].update_model(played_bids[cl], c)


class Step4UCBFixedClassesLearner(MultiClassLearner):
    """
    This class first plays with a single context and then, after a burn_in period, with some fixed class mapping.
    """

    def __init__(self, environment: envs.MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                 kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float, class_map: dict,
                 burn_in: int = 0, ucb_constant: float = 1):
        super().__init__(environment, bids, prices)
        # the initial map is a single context
        self.class_map = {p: 0 for p in environment.user_profiles}
        # and after the burn in it becomes the optimal one
        self.after_swap_class_map = class_map
        self.played_turns_to_swap = burn_in
        self.ucb_constant = ucb_constant
        self.kernel = kernel
        self.alpha = alpha
        self.beta = beta
        # the initial map is a single context
        self.class_map = {p: 0 for p in environment.user_profiles}
        # and after the burn in it becomes the optimal one
        self.after_swap_class_map = class_map
        # if burn_in = 0, at the first play this changes to the after_swap_class_map before playing any arm
        self.n_classes = len(set(self.class_map.values()))
        self.a_estimators = [est.BeUCB1Estimator(self.ps.shape[0], ucb_constant) for _ in range(self.n_classes)]
        self.n_estimators = [est.GPUCBEstimator(bids, kernel, alpha, beta) for _ in range(self.n_classes)]
        self.c_estimators = [est.GPUCBEstimator(bids, kernel, alpha, beta) for _ in range(self.n_classes)]

    def play_round(self):
        if self.history.played_rounds() == self.played_turns_to_swap:
            self.swap_from_single_context_to_given_map()
        alphas = np.zeros(self.n_classes)
        ps_t = np.zeros(self.n_classes)
        ps_t_ind = np.zeros(self.n_classes, dtype=int)

        for cl in range(self.n_classes):
            if self.a_estimators[cl].get_non_pulled_arms().shape[0] == 0:
                alphas_est = self.a_estimators[cl].provide_estimations()
                ps_t[cl], ps_t_ind[cl] = single_class_price_opt(self.ps, alphas_est, self.env.prod_cost)
                alphas[cl] = alphas_est[ps_t_ind[cl]]
            else:
                ps_t_ind[cl] = self.a_estimators[cl].get_non_pulled_arms()[0]
                ps_t[cl] = self.ps[ps_t_ind[cl]]
                alphas[cl] = 0.3

        n_est = np.array([self.n_estimators[cl].provide_estimations(lower_bound=False) for cl in range(self.n_classes)])
        c_est = np.array([self.c_estimators[cl].provide_estimations(lower_bound=True) for cl in range(self.n_classes)])

        xs_t, xs_t_ind = multi_class_bid_opt(self.xs, ps_t, alphas, n_est, c_est, self.env.prod_cost)
        results = self.play_and_save(self.class_map, xs_t, ps_t)
        results = self.step_results_to_estimated_classes(results, self.class_map)
        for cl in range(self.n_classes):
            n, q, c = results[cl]
            self.a_estimators[cl].update_estimations(ps_t_ind[cl], q, n)
            self.n_estimators[cl].update_model(xs_t[cl], n)
            self.c_estimators[cl].update_model(xs_t[cl], c)

    def swap_from_single_context_to_given_map(self):
        self.class_map = self.after_swap_class_map
        self.n_classes = len(set(self.class_map.values()))
        self.a_estimators = [est.BeUCB1Estimator(self.ps.shape[0], self.ucb_constant) for _ in range(self.n_classes)]
        self.n_estimators = [est.GPUCBEstimator(self.xs, self.kernel, self.alpha, self.beta)
                             for _ in range(self.n_classes)]
        self.c_estimators = [est.GPUCBEstimator(self.xs, self.kernel, self.alpha, self.beta)
                             for _ in range(self.n_classes)]
        if self.history.played_rounds() == 0:
            # nothing done, no model to initialize
            return
        # initialize the Be estimators
        for t in range(self.history.played_rounds()):
            for profile in self.env.user_profiles:
                cl = self.class_map[profile]
                self.a_estimators[cl].update_estimations(
                    list(self.ps).index(self.history.ps[profile][t]),
                    self.history.qs[profile][t], self.history.ns[profile][t]
                )
        # initialize the GP estimators
        played_bids = np.zeros((self.n_classes, self.history.played_rounds()))
        clicks = np.zeros((self.n_classes, self.history.played_rounds()))
        costs = np.zeros((self.n_classes, self.history.played_rounds()))
        for profile in self.env.user_profiles:
            cl = self.class_map[profile]
            # observe that we played the same bids for profiles that are in the same class
            played_bids[cl, :] = np.array(self.history.xs[profile])
            clicks[cl, :] += np.array(self.history.ns[profile])
            costs[cl, :] += np.array(self.history.cs[profile])
        for cl in range(self.n_classes):
            self.n_estimators[cl].update_model(played_bids[cl, :], clicks[cl, :])
            self.c_estimators[cl].update_model(played_bids[cl, :], costs[cl, :])


class Step4UCBRealClassesLearner(Step4UCBFixedClassesLearner):
    """
    The UCB learner that knows the true classes in advance
    """

    def __init__(self, environment: envs.MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                 kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float,
                 burn_in: int = 0, ucb_constant: float = 1):
        super().__init__(environment, bids, prices, kernel, alpha, beta, environment.class_map, burn_in, ucb_constant)


class Step4UCBOneClassLearner(Step4UCBFixedClassesLearner):
    def __init__(self, environment: envs.MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                 kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float, ucb_constant: float = 1):
        super().__init__(environment, bids, prices, kernel, alpha, beta, {p: 0 for p in environment.user_profiles},
                         0, ucb_constant)


class StatefulContextLearner(MultiClassLearner, ABC):
    def __init__(self, environment: envs.MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                 context_gen: StatefulContextGenerator, burn_in: int):
        super().__init__(environment, bids, prices)
        self.context_gen = context_gen
        self.burn_in = burn_in

    def play_round(self):
        played_rounds = self.history.played_rounds()
        if played_rounds > self.burn_in and played_rounds % 14 == 13:
            self._update_context()
        self._play_with_context()

    def _update_context(self):
        data = get_data_from_history(self.history)
        n_classes = self.context_gen.update_context(data)
        self._adapt_to_new_context(n_classes)

    def _initialize_estimators(self, a_estimators: list[Union[est.BeTSEstimator, est.BeUCB1Estimator]],
                               n_estimators: list[est.BaseGPEstimator],
                               c_estimators: list[est.BaseGPEstimator]):
        # the estimators are already instantiated, here their model is initialized
        n_classes = len(set(self.context_gen.class_map.values()))
        # initialize the Be estimators
        for t in range(self.history.played_rounds()):
            for profile in self.env.user_profiles:
                cl = self.context_gen.class_map[profile]
                a_estimators[cl].update_estimations(
                    list(self.ps).index(self.history.ps[profile][t]),
                    self.history.qs[profile][t], self.history.ns[profile][t]
                )
        # initialize the GP estimators
        played_bids = np.zeros((n_classes, self.history.played_rounds()))
        clicks = np.zeros((n_classes, self.history.played_rounds()))
        costs = np.zeros((n_classes, self.history.played_rounds()))
        for profile in self.env.user_profiles:
            cl = self.context_gen.class_map[profile]
            # observe that we played the same bids for profiles that are in the same class
            played_bids[cl, :] = np.array(self.history.xs[profile])
            clicks[cl, :] += np.array(self.history.ns[profile])
            costs[cl, :] += np.array(self.history.cs[profile])
        for cl in range(n_classes):
            n_estimators[cl].update_model(played_bids[cl, :], clicks[cl, :])
            c_estimators[cl].update_model(played_bids[cl, :], costs[cl, :])

    def _update_estimations(self, played_bids: np.ndarray, played_price_indices: np.ndarray, results: dict,
                            a_estimators: list[Union[est.BeTSEstimator, est.BeUCB1Estimator]],
                            n_estimators: list[est.BaseGPEstimator],
                            c_estimators: list[est.BaseGPEstimator]):
        results = self.step_results_to_estimated_classes(results, self.context_gen.class_map)
        for cl in range(len(set(self.context_gen.class_map.values()))):
            n, q, c = results[cl]
            a_estimators[cl].update_estimations(played_price_indices[cl], q, n)
            n_estimators[cl].update_model(played_bids[cl], n)
            c_estimators[cl].update_model(played_bids[cl], c)

    @abstractmethod
    def _play_with_context(self):
        pass

    @abstractmethod
    def _adapt_to_new_context(self, n_classes: int):
        pass


class Step4TSStatefulContextLearner(StatefulContextLearner):
    """
    A learner that uses the StatefulContextGenerator. It updates the context each two weeks.
    If a split is performed, that decision doesn't change in the following weeks.
    """

    def __init__(self, environment: envs.MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                 kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, rng: np.random.Generator,
                 context_gen: StatefulContextGenerator, burn_in: int):
        super().__init__(environment, bids, prices, context_gen, burn_in)
        self.kernel = kernel
        self.alpha = alpha
        self.rng = rng

        n_classes = len(set(self.context_gen.class_map.values()))
        self.a_estimators = [est.BeTSEstimator(self.ps.shape[0], self.rng) for _ in range(n_classes)]
        self.n_estimators = [est.GPTSEstimator(bids, kernel, alpha, rng) for _ in range(n_classes)]
        self.c_estimators = [est.GPTSEstimator(bids, kernel, alpha, rng) for _ in range(n_classes)]

    def _play_with_context(self):
        alphas_est, n_est, c_est = self._get_estimations()

        xs_t, xs_t_ind, ps_t, ps_t_ind = multi_class_opt(self.xs, self.ps, alphas_est, n_est, c_est, self.env.prod_cost)
        results = self.play_and_save(self.context_gen.class_map, xs_t, ps_t)
        self._update_estimations(xs_t, ps_t_ind, results, self.a_estimators, self.n_estimators, self.c_estimators)

    def _adapt_to_new_context(self, n_classes: int):
        self.a_estimators = [est.BeTSEstimator(self.ps.shape[0], self.rng) for _ in range(n_classes)]
        self.n_estimators = [est.GPTSEstimator(self.xs, self.kernel, self.alpha, self.rng) for _ in range(n_classes)]
        self.c_estimators = [est.GPTSEstimator(self.xs, self.kernel, self.alpha, self.rng) for _ in range(n_classes)]
        self._initialize_estimators(self.a_estimators, self.n_estimators, self.c_estimators)

    def _get_estimations(self):
        n_classes = len(set(self.context_gen.class_map.values()))
        alphas_est = np.array([self.a_estimators[cl].provide_estimations() for cl in range(n_classes)])
        n_est = np.array([self.n_estimators[cl].provide_estimations() for cl in range(n_classes)])
        c_est = np.array([self.c_estimators[cl].provide_estimations() for cl in range(n_classes)])
        return alphas_est, n_est, c_est


class Step4UCBStatefulContextLearner(StatefulContextLearner):
    """
    A multi-class learner that uses a context gen to determine the contex each two weeks
    after an initial "burn in" period.
    If a split is performed, the decision doesn't change in the following weeks.
    UCB estimators are used.
    """

    def __init__(self, environment: envs.MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                 kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float,
                 context_gen: StatefulContextGenerator, burn_in: int, ucb_constant=1):
        super().__init__(environment, bids, prices, context_gen, burn_in)
        self.kernel = kernel
        self.alpha = alpha
        self.beta = beta
        self.ucb_constant = ucb_constant
        n_classes = len(set(self.context_gen.class_map.values()))
        self.a_estimators = [est.BeUCB1Estimator(prices.shape[0], ucb_constant) for _ in range(n_classes)]
        self.n_estimators = [est.GPUCBEstimator(bids, kernel, alpha, beta) for _ in range(n_classes)]
        self.c_estimators = [est.GPUCBEstimator(bids, kernel, alpha, beta) for _ in range(n_classes)]

    def _play_with_context(self):
        n_classes = len(set(self.context_gen.class_map.values()))
        alphas = np.zeros(n_classes)
        ps_t = np.zeros(n_classes)
        ps_t_ind = np.zeros(n_classes, dtype=int)
        for cl in range(n_classes):
            if self.a_estimators[cl].get_non_pulled_arms().shape[0] == 0:
                alphas_est = self.a_estimators[cl].provide_estimations()
                ps_t[cl], ps_t_ind[cl] = single_class_price_opt(self.ps, alphas_est, self.env.prod_cost)
                alphas[cl] = alphas_est[ps_t_ind[cl]]
            else:
                ps_t_ind[cl] = self.a_estimators[cl].get_non_pulled_arms()[0]
                ps_t[cl] = self.ps[ps_t_ind[cl]]
                alphas[cl] = 0.3

        n_est, c_est = self._get_gp_estimations()
        xs_t, xs_t_ind = multi_class_bid_opt(self.xs, ps_t, alphas, n_est, c_est, self.env.prod_cost)
        results = self.play_and_save(self.context_gen.class_map, xs_t, ps_t)
        self._update_estimations(xs_t, ps_t_ind, results, self.a_estimators, self.n_estimators, self.c_estimators)

    def _adapt_to_new_context(self, n_classes: int):
        self.a_estimators = [est.BeUCB1Estimator(self.ps.shape[0], self.ucb_constant) for _ in range(n_classes)]
        self.n_estimators = [est.GPUCBEstimator(self.xs, self.kernel, self.alpha, self.beta) for _ in range(n_classes)]
        self.c_estimators = [est.GPUCBEstimator(self.xs, self.kernel, self.alpha, self.beta) for _ in range(n_classes)]
        self._initialize_estimators(self.a_estimators, self.n_estimators, self.c_estimators)

    def _get_gp_estimations(self):
        n_classes = len(set(self.context_gen.class_map.values()))
        n_est = np.array([self.n_estimators[cl].provide_estimations(lower_bound=False) for cl in range(n_classes)])
        c_est = np.array([self.c_estimators[cl].provide_estimations(lower_bound=True) for cl in range(n_classes)])
        return n_est, c_est
