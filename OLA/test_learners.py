import numpy as np
import environments as envs
import optimization_utilities as op
from OLA.multi_class_learners import MultiClassLearner
from environments import SingleClassHistory


class Step4ClairvoyantLearner(MultiClassLearner):
    """
    The clairvoyant multi-class learner
    """
    def __init__(self, environment: envs.MultiClassEnvironment, bids: np.ndarray, prices: np.ndarray):
        super().__init__(environment, bids, prices)
        n_values = np.array([self.env.n[c](self.xs) for c in range(self.env.classes_count())])
        c_values = np.array([self.env.c[c](self.xs) for c in range(self.env.classes_count())])
        a_values = np.array([self.env.a[c](self.ps) for c in range(self.env.classes_count())])
        best_bids, best_bids_ind, best_ps, best_ps_ind = op.multi_class_opt(self.xs, self.ps,
                                                                            a_values, n_values, c_values,
                                                                            environment.prod_cost)
        self.best_bids = best_bids
        self.best_bids_ind = best_bids_ind
        self.best_ps = best_ps
        self.best_ps_ind = best_ps_ind

    def play_round(self):
        self.play_and_save(self.env.class_map, self.best_bids, self.best_ps)

    def get_expected_reward(self):
        """
        The reward that we expect
        (I think that maybe the History computes it wrongly)
        :return:
        """
        n_values = np.array([self.env.n(self.xs, c) for c in range(self.env.classes_count())])
        c_values = np.array([self.env.c(self.xs, c) for c in range(self.env.classes_count())])
        a_values = np.array([self.env.a(self.ps, c) for c in range(self.env.classes_count())])
        best_bids, best_bids_ind, best_ps, best_ps_ind = op.multi_class_opt(self.xs, self.ps,
                                                                            a_values, n_values, c_values,
                                                                            self.env.prod_cost)
        reward = 0
        for c in range(self.env.classes_count()):
            reward += SingleClassHistory.reward(best_ps[c], a_values[c, best_ps_ind[c]], n_values[c, best_bids_ind[c]],
                                                c_values[c, best_bids_ind[c]], self.env.prod_cost)
        return reward

# now it seems that the rewards of the history are correct
