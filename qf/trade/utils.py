import numpy as np
import os
import pandas as pd

# Suggested Stress-Test Version
# utils.py
def check_if_tradable(quote_stats):
    try:
        return quote_stats["Edge"] > 10 and quote_stats["tradable"]
    except:
        return False
    
def get_stats(model_stats, quote_name):
    try:
        return model_stats.loc[quote_name]
    except:
        return None

def get_mse(quote_stats):
    try:
         return quote_stats["MSE"]
    except:
        return None

def get_model_stats(current_dir, filename):
    model_stats_file = os.path.join(current_dir, "test-results", filename)
    if not os.path.exists(model_stats_file):
        print(f"Error: Report file {filename} not found.")
        return None
        
    model_stats = pd.read_csv(model_stats_file)
    # Clean whitespace from column names if any
    model_stats.columns = model_stats.columns.str.strip()
    model_stats.set_index('Ticker', inplace=True)
    return model_stats

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
