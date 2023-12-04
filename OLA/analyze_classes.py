import numpy as np
import matplotlib.pyplot as plt
from environments import SingleClassEnvironmentHistory
import new_environment_properties as ep
import optimization_utilities as op

# verify that each class has a different optimum and plotting

xs = ep.get_bids()
ps = ep.get_prices()
# add prices to get a better plot (5 points are actually few)
expanded_ps = np.linspace(start=ps[0], stop=ps[-1])
prod_cost = ep.get_production_cost()

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 15))

best_rewards = []
for c in range(3):
    expanded_vs = ep.click_conversion_rate_multiclass(expanded_ps, c) * (expanded_ps - prod_cost)
    vs = ep.click_conversion_rate_multiclass(ps, c) * (ps - prod_cost)
    best_price = ps[np.argmax(vs)]
    best_value = np.max(vs)
    axes[0, c].plot(expanded_ps, expanded_vs)
    axes[0, c].scatter(ps, vs)
    axes[0, c].scatter(best_price, best_value, color='r')
    axes[0, c].set_title("Class " + str(c) + " price analysis")

    rewards = best_value * ep.daily_clicks_curve_multiclass(xs, c) - ep.click_cumulative_cost_multiclass(xs, c)
    axes[1, c].plot(xs, rewards)
    axes[1, c].scatter(xs[np.argmax(rewards)], np.max(rewards), color='r')
    axes[1, c].set_title("Class " + str(c) + " bid analysis")
    best_rewards.append(np.max(rewards))

# Now I ask what is the regret when I use the optimum of the first class for the other ones?

# best for class 0, not for the others
best_bid, _, best_price, _ = op.single_class_opt(xs, ps,
                                                 ep.click_conversion_rate_multiclass(ps, 0),
                                                 ep.daily_clicks_curve_multiclass(xs, 0),
                                                 ep.click_cumulative_cost_multiclass(xs, 0),
                                                 prod_cost)
for c in range(3):
    reward = SingleClassEnvironmentHistory.reward(best_price,
                                                  ep.click_conversion_rate_multiclass(best_price, c),
                                                  ep.daily_clicks_curve_multiclass(best_bid, c),
                                                  ep.click_cumulative_cost_multiclass(best_bid, c),
                                                  prod_cost)
    regret = best_rewards[c] - reward
    print("Regret class {} when using the optimum of class 0: {}".format(c, regret))

plt.show()
