import math
import numpy as np

OMEGA_MAX = 0.20
FLOOR =0.13
CEILING = 0.18

def calculate_levels(current_price, signal, delta_p, H, V, is_forex, is_jpy):
    cp = float(current_price)
    dp = float(delta_p)
    
    # Preserve Volume scaling logic
    v_factor = math.sqrt(max(0, float(V)))
    v_multiplier = 1.0 + min(v_factor, 2.0) 

    # Preserve experimental parameters
    SL_TINY_BASE = 0.40 
    TP_BASE = 1.0
    H_SCALAR = 7.5
    
    # Calculate Volatility-based distance
    vol_sl_distance = cp * dp * SL_TINY_BASE * v_multiplier
    
    # LOOPHOLE FIX: Structural Floor
    # Prevents sl_distance from becoming too small, which caps max leverage
    sl_floor_pct = 0.001 if is_forex else 0.005
    sl_floor_distance = cp * sl_floor_pct
    
    # Take the larger of the two to ensure a safe risk-denominator
    sl_distance = max(vol_sl_distance, sl_floor_distance)
    
    # Preserve Take Profit logic with Force (H) scaling
    tp_multiplier = (TP_BASE + (H_SCALAR * float(H))) * v_multiplier
    tp_distance = cp * dp * tp_multiplier

    tick_size = 0.01 if is_jpy else (0.0001 if is_forex else 0.01)

    if signal == 1:  # Long
        tp = current_price + tp_distance
        sl = current_price - sl_distance
        final_tp = math.floor(tp / tick_size) * tick_size
        final_sl = math.floor(sl / tick_size) * tick_size
    else:            # Short
        tp = current_price - tp_distance
        sl = current_price + sl_distance
        final_tp = math.ceil(tp / tick_size) * tick_size
        final_sl = math.ceil(sl / tick_size) * tick_size
        
    return float(final_tp), float(final_sl)
        
def update_hybrid_exit(position, row, delta_p, current_force, is_forex, buffered_floor):
    """
    Consolidated Exit Logic:
    1. Removes quantity loophole via Structural Risk Floor.
    2. Syncs multipliers with calculate_levels (SL_TINY_BASE = 0.40).
    3. Implements Dynamic Slippage based on volatility (dp).
    4. Preserves Early Stops (Momentum Decay & Break-even).
    """
    high = float(row['HIGH'])
    low = float(row['LOW'])
    close = float(row['CLOSE'])
    ticker = str(row['TICKER'])
    dp = float(delta_p)
    v_val = float(row['V'])
        
    is_jpy = "JPY" in ticker    
    tick_size = 0.01 if is_jpy else (0.0001 if is_forex else 0.01)

    # REFINEMENT: Dynamic Slippage & Fees
    # Scales slippage based on 10% of the current standard deviation (dp)
    slippage_cost = close * dp * 0.1 
    exit_fee_per_share = 0.005 if not is_forex else 0.0 
    profit_dist = (close - position.entry_price) * position.side

    def snap_to_tick(price, side):
        if side == 1: return math.floor(price / tick_size) * tick_size
        else: return math.ceil(price / tick_size) * tick_size


    if current_force < buffered_floor:
        # BENEFIT 1: Momentum Decay Exit (Early Stop)
        # If in profit and Force drops 30% from entry, exit immediately.
        momentum_trigger = (10 * tick_size) if is_forex else (position.entry_price * 0.001)
        if profit_dist > momentum_trigger and current_force < (position.entry_force * 0.70):
            exit_price_raw = close - (slippage_cost * position.side) 
            return None, snap_to_tick(exit_price_raw, position.side) - exit_fee_per_share, 0

        # BENEFIT 2: Break-even Protection
        # Locks in minor profit if the price moves significantly in our favor
        be_trigger = (6 * tick_size) if is_forex else (position.entry_price * 0.0006)
        if profit_dist > be_trigger:
            exit_price_raw = close - (slippage_cost * position.side)
            # Note: '0' maps to 'Early Stop' in your report dictionary
            return None, snap_to_tick(exit_price_raw, position.side) - exit_fee_per_share, 0

    # LOOPHOLE FIX: Synchronize Trailing Logic with Entry Floor
    # Use the exact same multipliers as calculate_levels
    v_multiplier = 1.0 + min(math.sqrt(max(0, v_val)), 2.0)
    sl_floor_pct = 0.001 if is_forex else 0.005
    
    # Calculate the minimum safe trailing distance
    vol_stop_dist = close * dp * 0.40 * v_multiplier
    floor_stop_dist = close * sl_floor_pct
    safe_trailing_dist = max(vol_stop_dist, floor_stop_dist)

    if position.side == 1:  # LONG
        # Check Hard Stops First
        if low <= position.stop_loss:
            return None, float(position.stop_loss) - exit_fee_per_share, -1
        elif high >= position.take_profit:
            return None, float(position.take_profit) - exit_fee_per_share, 1
            
        # Update Trailing Stop
        new_sl_raw = close - safe_trailing_dist
        # Optional: Aggressive trail if deep in profit
        if profit_dist > (dp * position.entry_price * 0.75):
            new_sl_raw = max(new_sl_raw, position.entry_price + (tick_size * 2))
            
        updated_sl = max(float(position.stop_loss), math.floor(new_sl_raw / tick_size) * tick_size)
        return position._replace(stop_loss=updated_sl), None, 0
            
    else:  # SHORT
        # Check Hard Stops First
        if high >= position.stop_loss:
            return None, float(position.stop_loss) - exit_fee_per_share, -1
        elif low <= position.take_profit:
            return None, float(position.take_profit) - exit_fee_per_share, 1
            
        # Update Trailing Stop
        new_sl_raw = close + safe_trailing_dist
        if profit_dist > (dp * position.entry_price * 0.75):
            new_sl_raw = min(new_sl_raw, position.entry_price - (tick_size * 2))
            
        updated_sl = min(float(position.stop_loss), math.ceil(new_sl_raw / tick_size) * tick_size)
        return position._replace(stop_loss=updated_sl), None, 0

def calculate_liquidity_cap(H, V, minimum):
    """
    H: Volume Wavelet Force (Geometric concentration).
    V: Squared standardized volume (Instability/Noise).
    minimum: The base lot size (anchor).
    
    Returns a capacity that scales the 'minimum' by the structural quality.
    """
    # H_ref is the force threshold (0.13) where we take the base lot.
    H_ref = 0.13
    
    # Geometric quality: Higher H and Lower V increase capacity.
    # We use sqrt(V) to get back to the Z-score magnitude scale.
    quality = (H / H_ref) / (1 + np.sqrt(V))
    
    return int(minimum * quality)

def calculate_dynamic_qty(confidence, H, V, current_capital, entry_price, stop_loss, is_forex, min_qty=1, max_qty=1000, max_risk_pct=0.02, max_cap_pct=0.03):
    """
    Compounds capital using Volatility-Targeting.
    Constrained by:
    1. A hard 2% Risk-of-Capital Rule (Stop Loss based)
    2. A hard Capital Exposure Limit (Notional based)
    """
    # 1. Calculate Absolute Stop Loss Distance (Volatility Risk)
    sl_distance = abs(entry_price - stop_loss)
    if sl_distance <= 0:
        sl_distance = entry_price * 0.001
        
    # 2. Performance-Based Conviction (Unchanged compounding logic)
    conviction = (confidence * 0.6) + (H * 0.2) + (abs(V) * 0.2) if H > 0 and V > 0 else confidence
    conviction = max(0.1, min(1.0, conviction))
    
    # 3. The Risk Constraint (Dollar Risk Calculation)
    # This limits how much is LOST if the stop is hit.
    target_risk_pct = max_risk_pct * conviction
    dollar_risk = current_capital * target_risk_pct
    
    # 4. Volatility-Adjusted Quantity
    qty_by_risk = int(dollar_risk / sl_distance)
    
    # 5. NEW: Capital Exposure Constraint (The "2% Capital Limit")
    # This limits how much is SPENT to open the position.
    # For FOREX, we still allow leverage (e.g., max_leverage of 5.0)
    # For Stocks, we cap it at your max_cap_pct (e.g., 0.02 for 2%)
    if is_forex:
        max_leverage = 5.0 
        max_allowed_notional = current_capital * max_leverage
    else:
        # This enforces your "min 2% capital limit" correctly
        max_allowed_notional = current_capital * max_cap_pct

    qty_by_cap = int(max_allowed_notional / entry_price)
    
    # 6. Apply all constraints
    # Takes the smaller of risk-based size or capital-limit-based size
    final_qty = min(qty_by_risk, qty_by_cap)
    
    # 7. LIQUIDITY CONSTRAINTS (Unchanged)
    if final_qty < min_qty:
        return 0
        
    final_qty = min(final_qty, max_qty)
    
    return final_qty