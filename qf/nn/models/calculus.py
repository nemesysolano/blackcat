import numpy as np
from scipy.optimize import brentq

def fractional_integral_weights(order, N):
    """Generates w_k for fractional integration: w_k = w_{k-1} * (k - 1 + s) / k"""
    weights = np.zeros(N)
    weights[0] = 1.0  # w_0 = 1
    for k in range(1, N):
        weights[k] = weights[k-1] * (k - 1 + order) / float(k)
    return weights

def fractional_integral(weights, values):
    """
    Calculates L_hat(t) = sum_{k=0}^{N-1} |w_k(-s)| * Lambda(t-k)
    """
    N = len(values)
    return np.dot(np.abs(weights), values)

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

def fractional_order(Λ, L, bracket=(1e-6, 1)):
    """
    Finds the fractional order 's' such that the dot product of 
    weights(s) and returns_history equals lambda_target.
    
    Args:
        lambda_target: The known Λ(t) value.
        returns_history: Array of [L(t), L(t-1), ..., L(t-N+1)].
        bracket: The search interval for s.
    """
    n = len(L)
    
    # Define the objective function f(s)
    def objective(s):
        weights = fractional_derivative_weights(s, n)
        # Λ(t) = sum(w_k * L(t-k))
        current_lambda = np.dot(weights, L)
        return current_lambda - Λ

    try:
        # Brent's method finds the root where objective(s) == 0
        order = brentq(objective, bracket[0], bracket[1])
        return order
    except ValueError:
        # This happens if the sign of f(a) and f(b) are the same
        # implying no root (or multiple roots) in the bracket.
        return None
