import math
import numpy as np
import math
import numpy as np

# --- STRATEGY CONSTANTS ---
MAX_LEVEL = 0.05       # 5% Hard Cap on Stop Loss (Fixes PTGX/NVTS blowouts)
MIN_LEVEL = 0.015      # 1.5% Minimum Stop (Prevents noise shakeouts)
TRAIL_START = 0.02     # Start trailing after 2% profit
TRAIL_DIST = 0.015     # Trail price by 1.5%
RISK_PER_TRADE = 0.02  # Risk 2% of capital per trade

# Fields available in df dataframe passed to `calculate_stock_levels`
# ----------------+------------------------------+
# TICKER          |AAPL                          |
# TIMESTAMP       |2016-02-22 05:00:00.000       |
# OPEN            |21.810000                     |
# HIGH            |21.950000                     |
# LOW             |21.720000                     |
# CLOSE           |21.940000                     |
# δP              |0.0001                        |
# V               |0.0000000000000000000000000000|
# AV              |137123200.000000000000        |
# H               |0.2488120027091331            |



def calculate_stock_levels(current_index, df, current_price, effective_dir, δf):
    """
    Calculates Stop Loss and Take Profit levels based on Volatility (V),
    but constrained by strict percentage caps to prevent large drawdowns.
    """
    row = df.iloc[current_index]
    
    # 1. Use Volatility (V) to gauge "noise" width
    # If V is missing or 0, default to a standard 2% width
    volatility = float(row['V']) if 'V' in row and row['V'] > 0 else 0.01
    
    # 2. Dynamic Stop Width: 
    # High Volatility -> Wider Stop (up to MAX_LEVEL)
    # Low Volatility  -> Tighter Stop (down to MIN_LEVEL)
    # We use a multiplier (e.g., 3x Volatility) to estimate a safe buffer
    raw_stop_dist = volatility * 3.0
    
    # 3. Apply Hard Clamps (The "Safety Net")
    stop_pct = max(MIN_LEVEL, min(raw_stop_dist, MAX_LEVEL))
    
    # 4. Target Ratio (2:1 Reward to Risk ideally, but at least 1.5:1)
    target_pct = stop_pct * 2.0
    
    # 5. Calculate Prices
    if effective_dir == 1: # LONG
        stop_loss = current_price * (1 - stop_pct)
        take_profit = current_price * (1 + target_pct)
    else: # SHORT
        stop_loss = current_price * (1 + stop_pct)
        take_profit = current_price * (1 - target_pct)
        
    return take_profit, stop_loss

def update_stock_position(current_step_index, df, position, δf):
    """
    Manages the active trade. Implements:
    1. Hard Stop/Take Profit checks.
    2. Trailing Stop to lock in peaks (Fixes PTGX giveback).
    3. Time-based "Stale Trade" exit (Fixes LDI fakeouts).
    """
    current_row = df.iloc[current_step_index]
    current_close = float(current_row['CLOSE'])
    current_low = float(current_row['LOW'])
    current_high = float(current_row['HIGH'])
    
    exit_price = None
    exit_reason = None
    exit_fee_per_share = 0.005 # Estimated slippage/commissions

    # --- A. TIME STOP (Livermore Rule) ---
    # If trade is > 5 bars old and still negative/flat, tighten stop to Break Even
    bars_held = current_step_index - position.entry_index
    if bars_held > 5:
        # Check if we are barely moving
        if position.side == 1 and current_close < position.entry_price * 1.005:
            # We are stalling. Tighten stop to just below current price to cut risk
            position = position._replace(stop_loss = max(position.stop_loss, current_close * 0.995))
        elif position.side == -1 and current_close > position.entry_price * 0.995:
            position = position._replace(stop_loss = min(position.stop_loss, current_close * 1.005))

    # --- B. TRAILING STOP LOGIC ---
    # Calculate the best price seen since entry (High for Longs, Low for Shorts)
    # Note: position.state contains history UP TO the previous step.
    past_highs = [s.high_price for s in position.state] if position.state else []
    past_lows = [s.low_price for s in position.state] if position.state else []
    
    if position.side == 1: # LONG
        # Get highest high seen during the trade (including today)
        best_price = max(past_highs + [current_high])
        
        # If we are in significant profit (e.g. > 2%), start trailing
        if best_price >= position.entry_price * (1 + TRAIL_START):
            # Move Stop Loss up to (Peak - Trail Distance)
            new_stop = best_price * (1 - TRAIL_DIST)
            # Never move stop down, only up
            if new_stop > position.stop_loss:
                position = position._replace(stop_loss = new_stop)
                
        # CHECK EXITS
        if current_low <= position.stop_loss:
            exit_price = position.stop_loss
            exit_reason = -1 # Or Trailing Stop
        elif current_high >= position.take_profit:
            exit_price = position.take_profit
            exit_reason = 1
            
    elif position.side == -1: # SHORT
        # Get lowest low seen
        best_price = min(past_lows + [current_low])
        
        # If we are in significant profit (price dropped > 2%)
        if best_price <= position.entry_price * (1 - TRAIL_START):
            # Move Stop Loss down to (Trough + Trail Distance)
            new_stop = best_price * (1 + TRAIL_DIST)
            # Never move stop up (loosening), only down (tightening)
            if new_stop < position.stop_loss:
                position= position._replace(stop_loss = new_stop)
        # CHECK EXITS
        if current_high >= position.stop_loss:
            exit_price = position.stop_loss
            exit_reason = -1
        elif current_low <= position.take_profit:
            exit_price = position.take_profit
            exit_reason = 1

    # Return structure expected by trade.py
    if exit_price is not None:
        return None, exit_price - exit_fee_per_share, exit_reason
    else:
        return position, None, 0

def calculate_stock_dynamic_qty(current_step_index, df, current_capital, entry_price, stop_loss, risk, δf):
    """
    Calculates position size based on Risk Percentage.
    Ensures we don't bet more than we can lose.
    """
    if current_capital <= 0 or entry_price <= 0:
        return 0

    # 1. Determine Risk Per Share
    risk_per_share = abs(entry_price - stop_loss)
    
    # Safety: If stop is somehow same as entry, default to a safe buffer
    if risk_per_share == 0:
        risk_per_share = entry_price * 0.02

    # 2. Determine Capital at Risk (e.g., $200 on a $10k account)
    amount_to_risk = current_capital * RISK_PER_TRADE
    
    # 3. Calculate Quantity
    qty = int(amount_to_risk / risk_per_share)
    
    # 4. Leverage Cap (Do not exceed 95% of available cash)
    max_capital_allocation = current_capital * 0.95
    if (qty * entry_price) > max_capital_allocation:
        qty = int(max_capital_allocation / entry_price)
        
    return qty


