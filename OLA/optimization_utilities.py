import numpy as np


def single_class_price_opt(prices: np.ndarray, estimated_alphas: np.ndarray):
    vs = estimated_alphas * prices * 100
    best_p_ind = np.argmax(vs)
    best_p = prices[best_p_ind]
    return best_p, best_p_ind


def single_class_bid_opt(bids: np.ndarray, price: float, alpha: float,
                         estimated_clicks: np.ndarray, estimated_costs: np.ndarray):
    objs = price * alpha * estimated_clicks * 100 - estimated_costs
    best_bid_ind = np.argmax(objs)
    best_bid = bids[best_bid_ind]
    return best_bid, best_bid_ind


def single_class_opt(bids: np.ndarray, prices: np.ndarray,
                     estimated_alphas: np.ndarray, estimated_clicks: np.ndarray, estimated_costs: np.ndarray):
    """
    Given the possible bids and prices and the estimated environment parameters,
    this function returns the best solution.
    Obs: for some steps, you use the actual values for some parameters and not the estimations,
    but the optimizer is the same
    :param bids: the available bids
    :param prices: the available prices
    :param estimated_alphas: the estimated conversion rates, estimated_alphas[i]= alpha for price=prices[i]
    :param estimated_clicks: the estimated number of clicks, estimated_clicks[i]= number of clicks for bid=bids[i]
    :param estimated_costs: the estimated advertising costs, estimated_costs[i]= number of clicks for bid=bids[i]
    :return: (bid, bid_index, price, price_index) the optimal bid and price to be played given the estimations
    and relative indices in the arrays bids and prices
    """

    best_p, best_p_ind = single_class_price_opt(prices, estimated_alphas)
    best_bid, best_bid_ind = single_class_bid_opt(bids, best_p, estimated_alphas[best_p_ind], estimated_clicks,
                                                  estimated_costs)
    return best_bid, best_bid_ind, best_p, best_p_ind


def multi_class_price_opt(prices: np.ndarray, estimated_alphas: np.ndarray):
    """
    :param prices: an array with the available prices
    :param estimated_alphas: a matrix where row c contains the alpha values estimated for class c
    :return: (ps,ps_ind)
    ps: np.ndarray contains the optimal prices (one for class)
    ps_ind: np.ndarray contains the optimal prices indices (according to the parameter prices)
    """
    # vs[class i] = estimated_alpha[class i] * prices
    vs = estimated_alphas * prices
    best_ps_ind = np.argmax(vs, axis=1)
    best_ps = prices[best_ps_ind]
    return best_ps, best_ps_ind


def multi_class_bid_opt(bids: np.ndarray, prices: np.ndarray, alphas: np.ndarray,
                        estimated_clicks: np.ndarray, estimated_costs: np.ndarray):
    """
    :param bids: the available bids
    :param prices: the (estimated) optimal prices, one for estimated class
    :param alphas: the (estimated) conversion rates with the optimal prices, one for class
    :param estimated_clicks: a matrix where row c contains the estimated number of clicks estimated for class c,
    shape=(n_classes,n_bids)
    :param estimated_costs: a matrix where row c contains the estimated advertising costs estimated for class c,
    shape=(n_classes,n_bids)
    :return: (bids, bids_ind)
    both array with the optimal bids and bids indices (one element for class)
    """
    vs = prices * alphas
    objs = estimated_clicks * vs.reshape((vs.shape[0], 1)).repeat(estimated_clicks.shape[1], axis=1) - estimated_costs
    best_bids_ind = np.argmax(objs, axis=0)
    best_bids = bids[best_bids_ind]
    return best_bids, best_bids_ind


def multi_class_opt(bids: np.ndarray, prices: np.ndarray,
                    estimated_alphas: np.ndarray, estimated_clicks: np.ndarray, estimated_costs: np.ndarray):
    best_ps, best_ps_ind = multi_class_price_opt(prices, estimated_alphas)
    best_bids, best_bids_ind = multi_class_bid_opt(bids, best_ps,
                                                   estimated_alphas[np.arange(estimated_alphas.shape[0]), best_ps_ind],
                                                   estimated_clicks, estimated_costs)
    return best_bids, best_bids_ind, best_ps, best_ps_ind
