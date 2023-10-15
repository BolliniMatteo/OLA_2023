import numpy as np
import matplotlib.pyplot as plt
import environment_properties as ep
import optimization_utilities as op

"""
Just scripts to find some "good" functions for the environments
"""

rng = np.random.default_rng(seed=3000)

c = 2

x = ep.get_bids()
p = ep.get_prices()
N = lambda x: ep.daily_clicks_curve_multiclass(x, c)
C = lambda x: ep.click_cumulative_cost_multiclass(x, c)
a = lambda p: ep.click_conversion_rate_multiclass(p,c)

n_values = N(x)
c_values = C(x)
a_values = a(p)

best_bid, best_bid_ind, best_p, best_p_ind = op.single_class_opt(x, p,
                                                                 a_values, n_values, c_values)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 15))

axes = axes.flatten()

axes[0].scatter(x, N(x), label='number clicks', s=0.5)
axes[0].plot(x, np.zeros(x.shape), label='Zero')

axes[1].scatter(x, C(x), label='payment', s=0.5)
axes[1].scatter(x, N(x)*x, label='max payment allowed', s=0.5) # I must require that c(x) <= n(x)*x, else I pay more than my bid allows
axes[1].plot(x, np.zeros(x.shape), label='Zero')

axes[2].scatter(p, a(p), label='conversion rate', s=5)
axes[2].plot(p, np.zeros(p.shape), label='Zero')

axes[3].scatter(x, N(x)+ep.click_curve_noise(rng, x.shape[0]), label='noisy number clicks', s=0.5)
axes[3].plot(x, np.zeros(x.shape), label='Zero')

axes[4].scatter(x, C(x)+ep.click_cumulative_cost_noise(rng,x.shape[0]), label='noisy payment', s=0.5)
axes[4].scatter(x, (N(x)+ep.click_curve_noise(rng, x.shape[0]))*x, label='noisy max payment', s=0.5)
axes[4].plot(x, np.zeros(x.shape), label='Zero')

# this is not really a plot, just showing two values
# in theory I could make a heatmap (rp,x)->r
axes[5].plot(x, np.zeros(x.shape) + best_p*a_values[best_p_ind]*n_values[best_bid_ind] - c_values[best_bid_ind], label='Optimal reward')
axes[5].plot(x, np.zeros(x.shape), label='Zero')

for i in range(len(axes)):
    axes[i].legend()

plt.show()
