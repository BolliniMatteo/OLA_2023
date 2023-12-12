import sklearn
from abc import ABC, abstractmethod

import environments as envs
import estimators as est
from OLA.optimization_utilities import *
from environments import SingleClassEnvironmentHistory

"""
The actual learners for the various step and the optimization functions that they use
A learner usually has to:
- instantiate the history object, computing the clairvoyant solution 
- instantiate and use the estimators that it needs
- at each round, get the estimations, optimize, play and update the estimators
"""


# these functions use np to optimize over multiple classes
# you can also optimize each class independently without using np parallelization,
# and probably you won't notice the difference


class SingleClassLearner(ABC):
    def __init__(self, environment: envs.SingleClassEnvironment, bids: np.ndarray, prices: np.ndarray):
        self.env = environment
        self.ps = prices
        self.xs = bids
        self.history = SingleClassEnvironmentHistory(self.env)

    def play_and_save(self, bid, price):
        n, q, c = self.env.perform_day(bid, price)
        self.history.add_step(bid, price, n, q, c)
        return n, q, c

    @abstractmethod
    def play_round(self):
        pass


class SingleClassLearnerNonStationary(ABC):
    def __init__(self, environment: envs.SingleClassEnvironmentNonStationary, bids: np.ndarray, prices: np.ndarray):
        self.env = environment
        self.ps = prices
        self.xs = bids
        self.history = envs.SingleClassEnvironmentNonStationaryHistory(environment)

    def play_and_save(self, bid, price):
        n, q, c = self.env.perform_day(bid, price)
        self.history.add_step(bid, price, n, q, c)
        return n, q, c

    @abstractmethod
    def play_round(self):
        pass


class Step1UCBLearner(SingleClassLearner, ABC):
    def __init__(self, environment: envs.SingleClassEnvironment,
                 bids: np.ndarray, prices: np.ndarray, c: float = 1):
        super().__init__(environment, bids, prices)
        self.estimator = est.BeUCB1Estimator(self.ps.shape[0], c)

    def play_round(self):
        n_est = self.env.N(self.xs)
        c_est = self.env.C(self.xs)
        if self.history.played_rounds() < self.ps.shape[0]:
            # we have not played each arm at least once
            # we could also use the +inf given by the estimator and optimize as usual...
            p_t = self.ps[self.history.played_rounds()]
            p_t_ind = self.history.played_rounds()
            # we have no data: we estimate 0.5 for the optimization
            # confirmed by experiments: lower alpha gives more regret, larger alpha gives more or less the same
            alpha = 0.5
            x_t, x_t_ind = single_class_bid_opt(self.xs, p_t, alpha, n_est, c_est, self.env.prod_cost)
        else:
            alphas_est = self.estimator.provide_estimations(lower_bound=False)
            x_t, x_t_ind, p_t, p_t_ind = single_class_opt(self.xs, self.ps, alphas_est, n_est, c_est,
                                                          self.env.prod_cost)
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
        x_t, x_t_ind, p_t, p_t_ind = single_class_opt(self.xs, self.ps, alphas_est, n_est, c_est, self.env.prod_cost)
        n, q, c = self.play_and_save(x_t, p_t)
        # ignore the warning, the argmax over the whole array is a single int and not an array
        self.estimator.update_estimations(p_t_ind, q, n)


class Step2UCBLearner(SingleClassLearner):
    def __init__(self, environment: envs.SingleClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                 kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float):
        super().__init__(environment, bids, prices)
        self.n_estimator = est.GPUCBEstimator(bids, kernel, alpha, beta)
        self.c_estimator = est.GPUCBEstimator(bids, kernel, alpha, beta)
        self.alphas = self.env.A(self.ps)
        # in theory, I could compute here the best price
        # and then optimize just the bid,
        # but I will simply use the single_class_opt function
        # and let it re-compute the constant optimal price at each round

    def play_round(self):
        alphas_est = self.alphas
        n_est = self.n_estimator.provide_estimations(lower_bound=False)
        c_est = self.c_estimator.provide_estimations(lower_bound=True)
        x_t, x_t_ind, p_t, p_t_ind = single_class_opt(self.xs, self.ps, alphas_est, n_est, c_est, self.env.prod_cost)
        n, _, c = self.play_and_save(x_t, p_t)
        self.n_estimator.update_model(x_t, n)
        self.c_estimator.update_model(x_t, c)


class Step2TSLearner(SingleClassLearner):
    def __init__(self, environment: envs.SingleClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                 kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, rng: np.random.Generator):
        super().__init__(environment, bids, prices)
        self.n_estimator = est.GPTSEstimator(bids, kernel, alpha, rng)
        self.c_estimator = est.GPTSEstimator(bids, kernel, alpha, rng)
        self.alphas = self.env.A(prices)
        # in theory, I could compute here the best price
        # and then optimize just the bid,
        # but I will simply use the single_class_opt function
        # and let it re-compute the constant optimal price at each round

    def play_round(self):
        alphas_est = self.alphas
        n_est = self.n_estimator.provide_estimations()
        c_est = self.c_estimator.provide_estimations()
        x_t, x_t_ind, p_t, p_t_ind = single_class_opt(self.xs, self.ps, alphas_est, n_est, c_est, self.env.prod_cost)
        n, _, c = self.play_and_save(x_t, p_t)
        self.n_estimator.update_model(x_t, n)
        self.c_estimator.update_model(x_t, c)


class Step3UCBLearner(SingleClassLearner):
    def __init__(self, environment: envs.SingleClassEnvironment, bids: np.ndarray, prices: np.ndarray,
                 kernel: sklearn.gaussian_process.kernels.Kernel, alpha: float, beta: float, c: float = 1):
        super().__init__(environment, bids, prices)
        self.a_estimator = est.BeUCB1Estimator(prices.shape[0], c)
        self.n_estimator = est.GPUCBEstimator(bids, kernel, alpha, beta)
        self.c_estimator = est.GPUCBEstimator(bids, kernel, alpha, beta)

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
            p_t, p_t_ind = single_class_price_opt(self.ps, alphas_est, self.env.prod_cost)
            alpha = alphas_est[p_t_ind]

        n_est = self.n_estimator.provide_estimations(lower_bound=False)
        c_est = self.c_estimator.provide_estimations(lower_bound=True)
        x_t, x_t_ind = single_class_bid_opt(self.xs, p_t, alpha, n_est, c_est, self.env.prod_cost)
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
        x_t, x_t_ind, p_t, p_t_ind = single_class_opt(self.xs, self.ps, alphas_est, n_est, c_est, self.env.prod_cost)
        n, q, c = self.play_and_save(x_t, p_t)
        # ignore the warning: the argmax over the whole array returns an int, not an array
        self.a_estimator.update_estimations(p_t_ind, q, n)
        self.n_estimator.update_model(x_t, n)
        self.c_estimator.update_model(x_t, c)


class Step5UCBLearner(SingleClassLearnerNonStationary):
    def __init__(self, environment: envs.SingleClassEnvironmentNonStationary,
                 bids: np.ndarray, prices: np.ndarray, c: float = 1):
        super().__init__(environment, bids, prices)
        self.estimator = est.BeUCB1Estimator(self.ps.shape[0], c)

    def play_round(self):
        n_est = self.env.N(self.xs)
        c_est = self.env.C(self.xs)
        if self.history.played_rounds() < self.ps.shape[0]:
            # we have not played each arm at least once
            # we could also use the +inf given by the estimator and optimize as usual...
            p_t = self.ps[self.history.played_rounds()]
            p_t_ind = self.history.played_rounds()
            # we have no data: we estimate 0.5 for the optimization
            alpha = 0.5
            x_t, x_t_ind = single_class_bid_opt(self.xs, p_t, alpha, n_est, c_est, self.env.prod_cost)
        else:
            alphas_est = self.estimator.provide_estimations(lower_bound=False)
            x_t, x_t_ind, p_t, p_t_ind = single_class_opt(self.xs, self.ps, alphas_est, n_est, c_est,
                                                          self.env.prod_cost)
        n, q, c = self.play_and_save(x_t, p_t)
        # ignore the warning, the argmax over the whole array is a single int and not an array
        self.estimator.update_estimations(p_t_ind, q, n)


class Step5UCBChangeDetectorLearner(SingleClassLearnerNonStationary):
    def __init__(self, environment: envs.SingleClassEnvironmentNonStationary,
                 bids: np.ndarray, prices: np.ndarray, c: float, burn_in_steps: int,
                 epsilon=0.05, max_num_changes=2,
                 time_horizon=365):
        super().__init__(environment, bids, prices)
        self.h = np.log(time_horizon / max_num_changes)
        self.c = c
        self.epsilon = epsilon
        self.cur_rand_walk_upper = 0
        self.cur_rand_walk_lower = 0
        self.mu_0 = 0
        self.estimator = est.BeUCB1Estimator(self.ps.shape[0], self.c)
        if burn_in_steps == -1:
            self.burn_in_steps = self.ps.shape[0]
        else:
            self.burn_in_steps = burn_in_steps
        self.rounds_since_last_detection = 0

    def play_round(self):
        n_est = self.env.N(self.xs)
        c_est = self.env.C(self.xs)
        # we first wait for the burn-in-phase to finish (we have a new burn-in every time there's a change)
        if self.rounds_since_last_detection < self.burn_in_steps:
            if self.rounds_since_last_detection >= self.ps.shape[0]:
                p_t = self.ps[self.rounds_since_last_detection]
                alpha = 0.5
                x_t, x_t_ind = single_class_bid_opt(self.xs, p_t, alpha, n_est, c_est, self.env.prod_cost)
            else:
                alphas_est = self.estimator.provide_estimations(lower_bound=False)
                x_t, x_t_ind, p_t, p_t_ind = single_class_opt(self.xs, self.ps, alphas_est, n_est, c_est,
                                                              self.env.prod_cost)
            self.play_and_save(x_t, p_t)
            # If we end the burn-in then we set everything up for the next round,
            # where the actual change detection starts
            if self.rounds_since_last_detection == (self.burn_in_steps - 1):
                start_interval = self.history.played_rounds() - self.rounds_since_last_detection
                end_interval = start_interval + self.burn_in_steps
                estimates = self.history.bernoulli_estimates()[start_interval:end_interval]
                self.mu_0 = np.mean(estimates)
                step_upper = estimates[-1] - self.mu_0 - self.epsilon
                step_lower = self.mu_0 - estimates[-1] - self.epsilon
                self.cur_rand_walk_lower = np.max(0, self.cur_rand_walk_lower + step_upper)
                self.cur_rand_walk_lower = np.max(0, self.cur_rand_walk_upper + step_lower)

        else:
            # play normally
            alphas_est = self.estimator.provide_estimations(lower_bound=False)
            x_t, x_t_ind, p_t, p_t_ind = single_class_opt(self.xs, self.ps, alphas_est, n_est, c_est,
                                                          self.env.prod_cost)
            self.play_and_save(x_t, p_t)
            # calculate steps
            last_reward = self.history.bernoulli_estimates()[-1]
            step_upper = last_reward - self.mu_0 - self.epsilon
            step_lower = self.mu_0 - last_reward - self.epsilon
            # update bounds
            self.cur_rand_walk_lower = np.max(0, self.cur_rand_walk_lower + step_upper)
            self.cur_rand_walk_lower = np.max(0, self.cur_rand_walk_upper + step_lower)
            # see if we identify a change after update (could be a false alarm...)
            if self.cur_rand_walk_lower > self.h or self.cur_rand_walk_upper > self.h:
                # change detectedS
                self.rounds_since_last_detection = -1
                # clear previous estimates
                self.estimator.reset_estimates()
        self.rounds_since_last_detection += 1


class Step5UCBWINLearner(SingleClassLearnerNonStationary):
    def __init__(self, environment: envs.SingleClassEnvironmentNonStationary,
                 bids: np.ndarray, prices: np.ndarray,
                 win_size: int, c: float = 1):
        super().__init__(environment, bids, prices)
        self.win_size = win_size
        self.c = c

    def play_round(self):
        n_est = self.env.N(self.xs)
        c_est = self.env.C(self.xs)
        t = self.history.played_rounds()
        # first round is t=0
        estimator = est.BeUCB1Estimator(self.ps.shape[0], self.c)
        # remember that t is excluded in the range()
        for i in range(max(0, t - self.win_size), t):
            p_ind = np.where(self.ps == self.history.ps[i])[0][0]
            estimator.update_estimations(p_ind, self.history.qs[i], self.history.ns[i])
        non_pulled_arms = estimator.get_non_pulled_arms()
        if non_pulled_arms.shape[0] != 0:
            p_t_ind = non_pulled_arms[0]
            p_t = self.ps[p_t_ind]
            # we have no data in our window
            # but if we don't find the arm in our window, then it wasn't a good arm in the last win_size days
            # we can assume that it has a not so large alpha value whn choosing the bid
            # this collects fewer samples for it, but avoid losing much reward
            # (with a large estimated alpha we tend to a larger bid, but we can get a negative reward
            # if we have many clicks and relative cost to pay, but few conversions to profit)
            x_t, _ = single_class_bid_opt(self.xs, p_t, 0.4, n_est, c_est, self.env.prod_cost)
        else:
            alphas_est = estimator.provide_estimations(lower_bound=False)
            x_t, _, p_t, _ = single_class_opt(self.xs, self.ps, alphas_est, n_est, c_est, self.env.prod_cost)
        self.play_and_save(x_t, p_t)


