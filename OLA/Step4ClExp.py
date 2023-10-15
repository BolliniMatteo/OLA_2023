import numpy as np
import sklearn.gaussian_process.kernels
import warnings
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import environment_properties as ep
from OLA.test_learners import Step4ClairvoyantLearner
from OLA.context_gen import ContextGeneration
from OLA.environments import MultiClassEnvironment
from OLA.simulators import simulate_multi_class, plot_multi_class_sim_result


# TODO: address the kernel warnings
warnings.filterwarnings("ignore")


def env_init_step4(rng: np.random.Generator):
    n_features = ep.n_features
    n_classes = ep.n_classes
    class_map = ep.class_map
    user_prob_map = ep.user_prob_map
    n = [lambda bid: ep.daily_clicks_curve_multiclass(bid, cl) for cl in range(n_classes)]
    en = lambda: ep.click_curve_noise(rng, 1)
    c = [lambda bid: ep.click_cumulative_cost_multiclass(bid, cl) for cl in range(n_classes)]
    ec = lambda: ep.click_cumulative_cost_noise(rng, 1)
    a = [lambda p: ep.click_conversion_rate_multiclass(p, cl) for cl in range(n_classes)]
    return MultiClassEnvironment(n_features, class_map, user_prob_map, n, en, c, ec, a, rng)


def main():
    seed = 5000
    bids = ep.get_bids()
    prices = ep.get_prices()
    rng = np.random.default_rng(seed=seed)
    env_init = lambda: env_init_step4(rng)

    learner_init = lambda env: Step4ClairvoyantLearner(env,bids,prices)

    T = 50  # TODO: should be 365
    n_runs = 1

    print("GP-UCB learner, clairvoyant", flush=True)
    results = simulate_multi_class(env_init, learner_init, T, n_runs)

    print("Expected reward: ", learner_init(env_init()).get_expected_reward())

    plot_multi_class_sim_result(results)


if __name__ == '__main__':
    main()