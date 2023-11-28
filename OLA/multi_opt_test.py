import numpy as np
import matplotlib.pyplot as plt
import environment_properties as ep
import optimization_utilities as op

"""
This file is sued to check whether the multi-class opt does the same thing as the single-class opt
(but more optimized thanks o the usage of matrices in numpy)
"""

x = ep.get_bids()
p = ep.get_prices()

n_values = np.array([ep.daily_clicks_curve_multiclass(x, c) for c in range(3)])
c_values = np.array([ep.click_cumulative_cost_multiclass(x, c) for c in range(3)])
a_values = np.array([ep.click_conversion_rate_multiclass(p, c) for c in range(3)])
print("Shapes: {}, {}, {}".format(n_values.shape, c_values.shape, a_values.shape))

best_bids, best_bids_ind, best_ps, best_ps_ind = op.multi_class_opt(x, p,
                                                                    a_values,
                                                                    n_values,
                                                                    c_values, 15)
for c in range(3):
    best_bid, best_bid_ind, best_p, best_p_ind = op.single_class_opt(x, p,
                                                                     a_values[c, :],
                                                                     n_values[c, :],
                                                                     c_values[c, :], 15)
    print("-----Class {}-----".format(c))
    print("best_bid: {}=={}".format(best_bids[c], best_bid))
    print("best_bid_ind: {}=={}".format(best_bids_ind[c], best_bid_ind))
    print("best_p: {}=={}".format(best_ps[c], best_p))
    print("best_p_ind: {}=={}".format(best_ps_ind[c], best_p_ind))
