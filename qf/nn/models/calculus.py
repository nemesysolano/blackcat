import numpy as np
from scipy.optimize import brentq
from qf.nativemath.impl import get_fractional_order as fractional_order


def fractional_derivative_weights(order, N):
    """
    Generates fractional differentiation weights w_k for a given order s.
    w_0 = 1
    w_k = w_{k-1} * (k - 1 - s) / k
    """
    weights = [1.0]
    for k in range(1, N):
        weights.append(weights[-1] * (k - 1 - order) / k)
    return np.array(weights)


def fractional_orders(Lambda, L):
    orders = []
    
    for i in range(len(Lambda)):
        t = L.index[i]
        orders.append(fractional_order(Lambda[i].astype(np.float64), L.loc[t].values.astype(np.float64)))
    return tuple(orders)    