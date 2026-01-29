import numpy as np

# Suggested Stress-Test Version
# utils.py

def dynamic_slippage(atr_pct, base_median_bps=3.5, base_sigma=1.1):
    """
    Simulates realistic friction including spread, fees, and impact.
    3.5 bps = 0.035% of price.
    """
    # Using a higher sigma (1.1) to simulate 'unlucky' fills/fat tails
    noise = np.random.lognormal(mean=np.log(base_median_bps), sigma=base_sigma)
    
    # Scale cost by volatility (ATR). 
    # If ATR% is high, the spread usually widens and fills are harder to get.
    # We increase the ceiling from 8.0 to 12.0
    turbulence_factor = np.clip(atr_pct / 0.005, 1.0, 12.0) 
    
    return (noise * turbulence_factor) / 10000

def random_slippage(): # just a random nummber between 0.01 and 0.10
    return np.random.uniform(0.01, 0.10)
    
def apply_integer_nudge(price, dist, is_tp, is_long):
    """
    Adjusts the target distance to avoid clustering exactly on integer levels.
    """
    target_price = price + dist if (is_long and is_tp) or (not is_long and not is_tp) else price - dist
    nudge = 0.0001 # Small offset to push past the integer
    
    # If the target is very close to an integer, nudge it
    if abs(target_price - round(target_price)) < 0.001:
        if (is_long and is_tp) or (not is_long and not is_tp):
            dist += nudge # Push TP further or SL wider
        else:
            dist -= nudge # Pull SL tighter or TP closer
    return dist
