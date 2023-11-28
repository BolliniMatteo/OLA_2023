import numpy as np
import matplotlib.pyplot as plt
from environments import SingleClassEnvironment
import new_environment_properties as ep
import optimization_utilities as op

"""
Just scripts to find some "good" functions for the environments
"""

rng = np.random.default_rng(seed=3000)

xs = ep.get_bids()
ps = ep.get_prices()
# add prices to get a better plot (5 points are actually few)
expanded_ps = np.linspace(start=ps[0], stop=ps[-1])

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

for c in range(3):
    axes[0][c].plot(expanded_ps, ep.click_conversion_rate_multiclass(expanded_ps, c))
    axes[0][c].scatter(ps, ep.click_conversion_rate_multiclass(ps, c))
    axes[0][c].set_title('Conversion rate class %d' % c)

    axes[1][c].plot(xs, ep.daily_clicks_curve_multiclass(xs, c))
    # scatter points with noise
    axes[1][c].scatter(xs, ep.daily_clicks_curve_multiclass(xs, c) + ep.daily_click_curve_noise(rng, xs.shape[0]))
    axes[1][c].set_title('Daily click class %d' % c)

    axes[2][c].plot(xs, ep.click_cumulative_cost_multiclass(xs, c))
    axes[2][c].scatter(xs,
                       ep.click_cumulative_cost_multiclass(xs, c) + ep.advertising_costs_curve_noise(rng, xs.shape[0]))
    # plot the theoretical limit: bid * clicks
    axes[2][c].plot(xs, ep.daily_clicks_curve_multiclass(xs, c) * xs)
    axes[2][c].set_title('Advertising costs class %d' % c)

# Obs: it's the environment's duty to put a threshold on the noisy clicks and costs to make sure that they don't go
# below 0

plt.show()

# obs: the "floor" in the number of clicks result in an "erratic" curve

# next plot: fix the best bid and show how the reward varies with different prices
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
axes = axes.flatten()

prod_cost = 15

expanded_vs = ep.click_conversion_rate(expanded_ps) * (expanded_ps - prod_cost)
vs = ep.click_conversion_rate(ps) * (ps - prod_cost)
axes[0].plot(expanded_ps, expanded_vs)
axes[0].scatter(ps, vs)
axes[0].scatter(ps[np.argmax(vs)], np.max(vs), color='r', label="best")
axes[0].set_title("Different values for fixed bid")

# next plot: fix the price and show the reward with different prices
best_price = ps[np.argmax(vs)]
rs = ep.daily_clicks_curve(xs) * ep.click_conversion_rate(best_price) * (
            best_price - prod_cost) - ep.click_cumulative_cost(xs)
axes[1].plot(xs, rs)
axes[1].scatter(xs[np.argmax(rs)],
                rs[np.argmax(rs)],
                color='r', label="best")
axes[1].set_title("Different rewards for fixed price")

plt.show()
