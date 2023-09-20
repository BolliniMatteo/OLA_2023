import numpy as np

from OLA.environments import SingleClassEnvironment
from OLA.base_learners import Step1UCBLearner
from OLA.base_learners import Step1TSLearner
import environment_properties as ep
from OLA.simulators import simulate_single_class
from OLA.simulators import plot_single_class_sim_result


def env_init_step5(rng: np.random.Generator):
    N = ep.daily_clicks_curve
    en = lambda: ep.click_curve_noise(rng, None) # size None returns a scalar, size 1 an array of a single element
    C = ep.click_cumulative_cost
    ec = lambda: ep.click_cumulative_cost_noise(rng, None)
    A = ep.click_conversion_rate_abrupt

    return SingleClassEnvironment(N, en, C, ec, A, rng)
