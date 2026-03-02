import numpy as np
from qf.nn.models.layers import FractionalDiffLayer

def fractional_integral_weights(order, N):
    """Generates w_k for fractional integration: w_k = w_{k-1} * (k - 1 + s) / k"""
    s = order
    weights = np.zeros(N)
    weights[0] = 1.0  # w_0 = 1
    for k in range(1, N):
        weights[k] = weights[k-1] * (float(k - 1) + s) / float(k)
    return weights

def fractional_integral(s, values):
    """
    Calculates L_hat(t) = sum_{k=0}^{N-1} |w_k(-s)| * Lambda(t-k)
    """
    N = len(values)
    weights = fractional_integral_weights(s, N)
    # Applying the discrete Riemann-Liouville approximation
    return np.dot(np.abs(weights), values)

def fractional_order(model):
    """
    Extracts the learned fractional order 's' from a trained 
    fractional price acceleration model.
    
    Returns:
        float: The learned value of s (constrained between 0.01 and 0.99)
    """
    for layer in model.layers:
        if isinstance(layer, FractionalDiffLayer):
            # layer.get_weights() returns a list of numpy arrays
            # The first element is our 's_order' weight
            s_value = layer.get_weights()[0][0]
            return float(s_value)
    
    raise ValueError("The provided model does not contain a FractionalDiffLayer.")