import numpy as np
import matplotlib.pyplot as plt
from environments import SingleClassEnvironment, SingleClassHistory
import new_environment_properties as ep
import optimization_utilities as op

"""
This analysis shows the curve that step 1 is optimizing
and the best reward that it can obtain with each price.
Observe that the best reward is achieved when both:
- the best price is chosen
- the estimation of the corresponding alpha is good enough
This is because the choice of the bid depend on the choice of the price and the estimated alpha

"""

xs = ep.get_bids()
ps = ep.get_prices()
# add prices to get a better plot (5 points are actually few)
expanded_ps = np.linspace(start=ps[0], stop=ps[-1])
prod_cost = ep.get_production_cost()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
axes = axes.flatten()

axes[0].set_title("Expected revenue per click (alpha*margin)")
axes[0].plot(expanded_ps, (expanded_ps - prod_cost) * ep.click_conversion_rate(expanded_ps))
axes[0].scatter(ps, (ps - prod_cost) * ep.click_conversion_rate(ps))
axes[0].scatter(ps[3], (ps[3] - prod_cost) * ep.click_conversion_rate(ps[3]), color='r')

expanded_best_bids = [op.single_class_bid_opt(xs, price,
                                              ep.click_conversion_rate(price),
                                              ep.daily_clicks_curve(xs),
                                              ep.click_cumulative_cost(xs),
                                              prod_cost)[0] for price in expanded_ps]
expanded_best_bids = np.array(expanded_best_bids)
best_bids = [op.single_class_bid_opt(xs, price,
                                     ep.click_conversion_rate(price),
                                     ep.daily_clicks_curve(xs),
                                     ep.click_cumulative_cost(xs),
                                     prod_cost)[0] for price in ps]
best_bids = np.array(best_bids)
expanded_rewards = SingleClassHistory.reward(expanded_ps,
                                             ep.click_conversion_rate(expanded_ps),
                                             ep.daily_clicks_curve(expanded_best_bids),
                                             ep.click_cumulative_cost(expanded_best_bids),
                                             prod_cost)
rewards = SingleClassHistory.reward(ps,
                                    ep.click_conversion_rate(ps),
                                    ep.daily_clicks_curve(best_bids),
                                    ep.click_cumulative_cost(best_bids),
                                    prod_cost)

axes[1].set_title("Best reward for price (with optimal bid)")
axes[1].plot(expanded_ps, expanded_rewards)
axes[1].scatter(ps, rewards)
axes[1].scatter(ps[3], rewards[3], color='r')

plt.show()
