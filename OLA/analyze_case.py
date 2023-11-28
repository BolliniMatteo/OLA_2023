import numpy as np
import matplotlib.pyplot as plt
from environments import SingleClassEnvironment
import new_environment_properties as ep
import optimization_utilities as op

xs = ep.get_bids()
ps = ep.get_prices()
# add prices to get a better plot (5 points are actually few)
expanded_ps = np.linspace(start=ps[0], stop=ps[-1])
prod_cost = 15

# fix the price and show the reward with different prices
price = 50
rs = ep.daily_clicks_curve(xs) * ep.click_conversion_rate(price) * (
            price - prod_cost) - ep.click_cumulative_cost(xs)
plt.plot(xs, rs)
plt.scatter(xs[np.argmax(rs)],
            rs[np.argmax(rs)],
            color='r', label="best")
plt.title("Different rewards for fixed price")
plt.show()
