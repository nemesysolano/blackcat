import numpy as np
from scipy.stats import geninvgauss

def random_gh(λ, α, β, δ, μ, N=50_000_000):
    """
    Generates random numbers from the Generalized Hyperbolic Distribution
    using a normal variance-mean mixture.
    """
    # 1. Calculate γ (gamma)
    # The condition α > |β| must be strictly maintained to avoid complex numbers.
    γ = np.sqrt(α**2 - β**2)
    
    # 2. Map standard quantitative parameters to SciPy's geninvgauss parameters
    # SciPy uses p and b, where the PDF is parameterized differently than standard texts.
    scipy_p = λ
    scipy_b = δ * γ
    scipy_scale = δ / γ
    
    # 3. Generate Generalized Inverse Gaussian (GIG) mixing variables (W)
    # This executes heavily in SciPy's compiled C-backend.
    W = geninvgauss.rvs(p=scipy_p, b=scipy_b, scale=scipy_scale, size=N)
    
    # 4. Generate Standard Normal variables (Z)
    Z = np.random.standard_normal(size=N)
    
    # 5. Assemble the Variance-Mean Mixture
    # Vectorized element-wise operation: X = μ + βW + √(W)Z
    X = μ + (β * W) + (np.sqrt(W) * Z)
    
    return X

# Example execution:
# X_samples = random_gh(λ=-0.5, α=1.5, β=0.0, δ=1.0, μ=0.0, N=50_000_000)