import numpy as np
import matplotlib.pyplot as plt
import new_environment_properties as ep

"""
Script to plot the curves of the three classes
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
    # axes[1][c].scatter(xs, ep.daily_clicks_curve_multiclass(xs, c) + ep.daily_click_curve_noise(rng, xs.shape[0]))
    axes[1][c].set_title('Daily clicks class %d' % c)

    axes[2][c].plot(xs, ep.click_cumulative_cost_multiclass(xs, c))
    axes[2][c].set_title('Daily advertising costs class %d' % c)
    # axes[2][c].scatter(xs,
    #                    ep.click_cumulative_cost_multiclass(xs, c) + ep.advertising_costs_curve_noise(rng, xs.shape[0]))
    # plot the theoretical limit: bid * clicks
    # axes[2][c].plot(xs, ep.daily_clicks_curve_multiclass(xs, c) * xs)
    # axes[2][c].set_title('Advertising costs class %d' % c)
# Obs: it's the environment's duty to put a threshold on the noisy clicks and costs to make sure that they don't go
# below 0 due to the noise
plt.show()
