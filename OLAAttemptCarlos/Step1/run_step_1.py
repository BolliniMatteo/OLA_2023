import matplotlib.pyplot as plt

from configs.plot_styling import *
from SingleClassPricingEnvironment import *
from TSLearner import *
from UCB1Learner import *

if __name__ == '__main__':
    # Parameters of Bernoulli distributions of each arm
    distributions = np.array([0.1, 0.15, 0.2, 0.25, 0.35])
    n_arms = distributions.size
    # Parameter (and average reward) of the best arm
    optimal_reward = distributions[4]

    T = 365
    number_of_experiments = 300
    ucb1_rewards_per_experiment = []
    ts_rewards_per_experiment = []

    for e in range(0, number_of_experiments):
        # Reset the environment and learners at each experiment
        # P.S. Resetting the environment is not necessary in this case
        environment = SingleClassPricingEnvironment(n_arms=n_arms, probabilities=distributions)
        ucb1_learner = UCB1Learner(n_arms=n_arms)
        ts_learner = TSLearner(n_arms=n_arms)

        for t in range(0, T):
            # UCB1 learner
            pulled_arm = ucb1_learner.pull_arm()
            reward = environment.round(pulled_arm)
            ucb1_learner.update(pulled_arm, reward)

            # TS learner
            pulled_arm = ts_learner.pull_arm()
            reward = environment.round(pulled_arm)
            ts_learner.update(pulled_arm, reward)

        ucb1_rewards_per_experiment.append(ucb1_learner.collected_rewards)
        ts_rewards_per_experiment.append(ts_learner.collected_rewards)

    plt.figure(0)
    plt.ylabel('Regret')
    plt.xlabel('t')
    plt.plot(np.cumsum(np.mean(optimal_reward - ucb1_rewards_per_experiment, axis=0)))
    plt.plot(np.cumsum(np.mean(optimal_reward - ts_rewards_per_experiment, axis=0)))
    plt.legend(['UCB1', 'TS'])
    plt.show()
