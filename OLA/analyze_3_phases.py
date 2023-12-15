import numpy as np
import matplotlib.pyplot as plt
import new_environment_properties as ep

# verify that the 5 phases for step 6 have ach one a different optimum

xs = ep.get_bids()
ps = ep.get_prices()
# add prices to get a better plot (5 points are actually few)
expanded_ps = np.linspace(start=ps[0], stop=ps[-1])
prod_cost = ep.get_production_cost()

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 15))

best_prices = []
ts = ep.get_three_phases_moments()

for i in range(3):
    expanded_vs = ep.conversion_rate_three_phases(expanded_ps, ts[i]) * (expanded_ps - prod_cost)
    vs = ep.conversion_rate_three_phases(ps, ts[i]) * (ps - prod_cost)
    axes[0, i].plot(expanded_ps, ep.conversion_rate_three_phases(expanded_ps, ts[i]))
    axes[0, i].scatter(ps, ep.conversion_rate_three_phases(ps, ts[i]))
    axes[0, i].scatter(ps[np.argmax(vs)], ep.conversion_rate_three_phases(ps[np.argmax(vs)], ts[i]), color='r')
    axes[0, i].set_title("Phase " + str(i) + " conversion rate")

    axes[1, i].plot(expanded_ps, expanded_vs)
    axes[1, i].scatter(ps, vs)
    axes[1, i].scatter(ps[np.argmax(vs)], np.max(vs), color='r')
    axes[1, i].set_title("Phase " + str(i) + " value")
    best_prices.append(ps[np.argmax(vs)])

best_prices.sort()

print("Sorted best prices:", best_prices)

plt.show()
