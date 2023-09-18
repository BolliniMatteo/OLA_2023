import numpy as np
import matplotlib.pyplot as plt
from environments import SingleClassEnvironment
import environment_properties as ep

"""
Just scripts to find some "good" functions for the environments
"""

rng = np.random.default_rng(seed=3000)

x = ep.get_bids()
p = ep.get_prices()
N = ep.daily_clicks_curve
C = ep.click_cumulative_cost
a = ep.click_conversion_rate

n_values = N(x)
c_values = C(x)
a_values = a(p)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 15))

axes = axes.flatten()

axes[0].scatter(x, N(x), label='number clicks', s=0.5)
axes[0].plot(x, np.zeros(x.shape), label='Zero')

axes[1].scatter(x, C(x), label='payment', s=0.5)
axes[1].scatter(x, N(x)*x, label='max payment', s=0.5)
axes[1].plot(x, np.zeros(x.shape), label='Zero')

axes[2].scatter(p, a(p), label='conversion rate', s=5)
axes[2].plot(p, np.zeros(p.shape), label='Zero')

axes[3].scatter(x, N(x)+ep.click_curve_noise(rng, x.shape[0]), label='noisy number clicks', s=0.5)
axes[3].plot(x, np.zeros(x.shape), label='Zero')

axes[4].scatter(x, C(x)+ep.click_cumulative_cost_noise(rng,x.shape[0]), label='noisy payment', s=0.5)
axes[4].scatter(x, (N(x)+ep.click_curve_noise(rng, x.shape[0]))*x, label='noisy max payment', s=0.5)
axes[4].plot(x, np.zeros(x.shape), label='Zero')

for i in range(len(axes)):
    axes[i].legend()

plt.show()
