import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

if __name__ == '__main__':
    rewards_per_arm = [
        [3, 4, 3, 1, 0],
        [2, 4, 5],
        [2, 1, 3, 3],
        [1, 6]
    ]

    mean_vector = [np.mean(arm_rewards, axis=0) for arm_rewards in rewards_per_arm]
    length_vector = [len(rewards_per_arm[i]) for i in range(len(rewards_per_arm))]
    print(length_vector)
    conf_term = np.sqrt(10 / np.array(length_vector))
    print(conf_term)
    print(mean_vector)
    print(conf_term + mean_vector)
    idx = np.argmax(conf_term + mean_vector)
    print(idx)

