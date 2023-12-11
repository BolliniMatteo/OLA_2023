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
    best_bid = xs[np.argmax(rewards)]
    axes[1, c].plot(xs, rewards)
    axes[1, c].scatter(best_bid, np.max(rewards), color='r')
    axes[1, c].set_title("Class " + str(c) + " bid analysis")
    best_rewards.append(np.max(rewards))

# Now I ask what is the regret when I merge the three classes in a single context?
# I can sum costs and clicks, but what about the conversion rate?
# It depends on how many users are of a class and how many of another, but that depends on the bid...

# Well, let's optimize by brute force
rewards = np.zeros((xs.shape[0], ps.shape[0]))
for i in range(xs.shape[0]):
    for j in range(ps.shape[0]):
        x = xs[i]
        p = ps[j]
        for cl in range(3):
            rewards[i, j] += SingleClassEnvironmentHistory.reward(p, ep.click_conversion_rate_multiclass(p, cl),
                                                                  ep.daily_clicks_curve_multiclass(x, cl),
                                                                  ep.click_cumulative_cost_multiclass(x, cl),
                                                                  ep.get_production_cost())
# That's the optimal reward when you play the same price and bid for every class
print("Optimal reward when mixing classes:", np.max(rewards))

plt.show()
