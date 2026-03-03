import math
import numpy as np
from qf.trade.fracdiff.signal import STALL, STRONG_BULLISH, STRONG_BEARISH, MEAN_REVERSION_LONG, MEAN_REVERSION_SHORT
from qf.trade.fracdiff.stats import FracDiffPosition, FracDiffTransaction
MAX_RISK_PER_TRADE = 0.03

def calculate_levels(current_index, signal, trade_dataset, L, L_hat, Λ, Λ_hat_t):
    """
    Returns (take_profit, stop_loss) based on Quantum Resonance.
    
    Logic:
    - Potential Barrier (v_b) is derived from the High-Low spread at current_index.
    - Acceleration (Λ) and Prediction (Λ_hat_t) determine the 'Tunneling' distance.
    """
    if signal == 0: # STALL
        return 0.0, 0.0, 0
    t = trade_dataset.index[current_index]
    current_price = trade_dataset.loc[t, 'CLOSE']
    high = trade_dataset.loc[t, 'HIGH']
    low = trade_dataset.loc[t, 'LOW']
    signal_direction = 0

    # 1. Define the Potential Barrier Width (v_b)
    # This represents the local 'energy' required to move the price particle
    local_range = high - low
    
    # Floor the buffer at 0.5% of price to ensure we aren't trapped in a zero-volatility calculation
    v_b = max(local_range, current_price * 0.005)

    # 2. Dynamic TP Expansion (Quantum Tunneling)
    # We use Λ_hat_t (forecasted force) to scale how far we expect the price to travel.
    # Higher predicted acceleration = deeper tunneling into the next price level.
    tunneling_factor = 1.0 + abs(Λ_hat_t * 10)

    # 3. Level Mapping
    if signal == STRONG_BULLISH:
        # Trend: Expecting a breakout/tunneling event
        take_profit = current_price + (v_b * 2.0 * tunneling_factor)
        stop_loss = current_price - (v_b * 1.5)
        signal_direction = 1
        
    elif signal == STRONG_BEARISH:
        # Trend: Expecting a breakdown
        take_profit = current_price - (v_b * 2.0 * tunneling_factor)
        stop_loss = current_price + (v_b * 1.5)
        signal_direction = -1


    elif signal == MEAN_REVERSION_LONG:
        # Trapped: Expecting a bounce back to the mean (L=0)
        # Targeted TP is tighter (1.0x buffer) because we are trading 'inside' the well
        take_profit = current_price + (v_b * 1.0)
        stop_loss = current_price - (v_b * 1.0)
        signal_direction = 1


    elif signal == MEAN_REVERSION_SHORT:
        # Trapped: Expecting a drop back to the mean
        take_profit = current_price - (v_b * 1.0)
        stop_loss = current_price + (v_b * 1.0)
        signal_direction = -1
    
    else:
        return 0.0, 0.0, signal_direction

    return round(take_profit, 4), round(stop_loss, 4), signal_direction

def try_early_close(current_index, position, next_open_price, exit_date):
    # Bug fix: Ensure profit_loss is initialized for the final check
    profit_loss = None
    if position.side == 1 and (next_open_price < position.stop_loss or next_open_price > position.take_profit):
        profit_loss = (next_open_price - position.entry_price) * position.quantity * position.side 
    elif position.side == -1 and (next_open_price > position.stop_loss or next_open_price < position.take_profit):
        profit_loss = (next_open_price - position.entry_price) * position.quantity * position.side

    if profit_loss is not None:            
        exit_reason = 1 if profit_loss > 0 else -1
        # Returns (None, transaction) to signify the position is closed
        return None, FracDiffTransaction.from_position(position, current_index, next_open_price, exit_reason, profit_loss, exit_date)
    return position, None # Ensure it returns position if no exit

def try_normal_close(current_index, position, low_price, high_price, exit_date):
    exit_reason = 0
    
    # --- NEW: Dynamic SL Tightening (The "No-Turn-Back" Rule) ---
    tp_distance = abs(position.take_profit - position.entry_price)
    progress = abs(high_price - position.entry_price) / tp_distance if position.side == 1 else \
               abs(position.entry_price - low_price) / tp_distance
    
    # If we reached 75% of TP, lock in at least 50% of the profit
    if progress >= 0.75:
        lock_in_price = position.entry_price + (position.side * tp_distance * 0.5)
        if position.side == 1:
            position = position._replace(stop_loss=max(position.stop_loss, lock_in_price))
        else:
            position = position._replace(stop_loss=min(position.stop_loss, lock_in_price))
    
    # Standard Trailing Stop (Break-even at 50% progress)
    elif progress >= 0.50:
        new_sl = max(position.stop_loss, position.entry_price) if position.side == 1 else min(position.stop_loss, position.entry_price)
        position = position._replace(stop_loss=new_sl)

    # Standard SL/TP Check
    if position.side == 1:
        if low_price < position.stop_loss: exit_reason = -1
        elif high_price > position.take_profit: exit_reason = 1
    elif position.side == -1:
        if high_price > position.stop_loss: exit_reason = -1
        elif low_price < position.take_profit: exit_reason = 1

    if exit_reason != 0:
        actual_exit_price = position.stop_loss if exit_reason == -1 else position.take_profit
        profit_loss = (actual_exit_price - position.entry_price) * position.quantity * position.side
        return None, FracDiffTransaction.from_position(position, current_index, actual_exit_price, exit_reason, profit_loss, exit_date)
        
    return position, None

def update_position(current_index, signal, trade_dataset, L, L_hat, Λ, Λ_hat_t, position):        
    """
    Updates the active position with high-frequency exit logic to minimize roundtrips.
    Includes: Signal Inversion, Stall Exhaustion, Resonance Decay, and Trailing Stops.
    """
    if position is None: 
        return None, None
    
    transaction = None
    exit_date = trade_dataset.index[current_index]
    bars_held = current_index - position.entry_index
    
    # --- 1. SIGNAL INVERSION EXIT (Regime Change) ---
    # If the force has flipped entirely against our position, exit at current price.
    is_inversion = (position.side == 1 and signal in [STRONG_BEARISH, MEAN_REVERSION_SHORT]) or \
                   (position.side == -1 and signal in [STRONG_BULLISH, MEAN_REVERSION_LONG])
                   
    # --- 2. RESONANCE DECAY EXIT (Alpha Exhaustion) ---
    # We check the entry force stored in the first state of the position history.
    entry_state = position.state[0]
    entry_force = abs(entry_state.Λ) + abs(entry_state.Λ_hat)
    current_force = abs(Λ) + abs(Λ_hat_t)
    
    # If the 'Quantum Push' has decayed below 15% of entry strength after 5 bars.
    is_force_decay = (bars_held >= 5 and current_force < (entry_force * 0.15))

    # --- 3. STALL EXHAUSTION (Time Decay) ---
    # If the model stays in STALL for too long, the predictive edge is likely gone.
    is_stall_exhaustion = (bars_held >= 10 and signal == STALL)

    # Execute Strategy-Based Exit
    if is_inversion or is_force_decay or is_stall_exhaustion:
        exit_price = trade_dataset.loc[exit_date, 'CLOSE']
        profit_loss = (exit_price - position.entry_price) * position.quantity * position.side
        # Categorize exit reason based on PnL at the moment of signal death
        exit_reason = 1 if profit_loss > 0 else -1 
        
        transaction = FracDiffTransaction.from_position(
            position, current_index, exit_price, exit_reason, profit_loss, exit_date
        )
        return None, transaction

    # --- 4. GAPS / EARLY CLOSE ---
    # Check for overnight gaps that might have blown past SL/TP
    if current_index == position.entry_index and current_index < len(trade_dataset) - 1:
        next_date = trade_dataset.index[current_index + 1]
        next_open_price = trade_dataset.loc[next_date, 'OPEN']
        position, transaction = try_early_close(current_index, position, next_open_price, next_date)

    # --- 5. DYNAMIC SL & INTRADAY EXITS ---
    # This handles the Trailing Stop and 75%-Progress No-Turn-Back rule
    if transaction is None:
        low_price = trade_dataset.loc[exit_date, 'LOW']
        high_price = trade_dataset.loc[exit_date, 'HIGH']
        position, transaction = try_normal_close(current_index, position, low_price, high_price, exit_date)
   
    return position, transaction

def calculate_stock_qty(current_index, trade_dataset, entry_price, stop_loss, current_capital, L, L_hat, Λ, Λ_hat, signal_direction):
    """
    Calculates quantity based on Risk-at-Risk (3% of capital) and 
    scales the conviction based on the Magnitude of the Resonance.
    """
    if stop_loss == 0 or entry_price == stop_loss:
        return 0

    # 1. Basic Risk Management (The 'Potential Barrier' width)
    # How much money are we willing to lose on this specific trade?
    cash_risk = current_capital * MAX_RISK_PER_TRADE
    
    # How much loss is incurred per single share?
    price_risk_per_share = abs(entry_price - stop_loss)
    
    # Base quantity: Risk-Parity approach
    base_qty = cash_risk / price_risk_per_share

    # 2. Conviction Scaling (Quantum Resonance Magnitude)
    # We look at the total 'Energy' of the signal. 
    # If the combined forces (Actual + Predicted) are strong, we move toward 100% of base_qty.
    # If the forces are weak (low resonance), we scale down to avoid noise.
    total_force_magnitude = abs(Λ) + abs(Λ_hat)
    
    # We use a tanh-like saturation to ensure scaling stays between 0.5 and 1.2
    # This prevents the model from taking dangerously large positions in high-volatility gaps.
    conviction_weight = 0.5 + (np.tanh(total_force_magnitude * 5) * 0.7)
    
    final_qty = base_qty * conviction_weight

    # 3. Liquidity & Capital Constraints
    # Ensure we don't try to buy more than we have cash for
    max_affordable = (current_capital * 0.95) / entry_price # Keep 5% for fees/slippage
    
    qty = min(final_qty, max_affordable)

    return int(math.floor(qty))

def create_position(quote_name, current_index, signal, trade_dataset, L, L_hat, Λ, Λ_hat, current_capital):
    t = trade_dataset.index[current_index]
    take_profit, stop_loss, signal_direction = calculate_levels(current_index, signal, trade_dataset, L, L_hat, Λ, Λ_hat)
    entry_price = trade_dataset.loc[t, 'CLOSE']
    qty = calculate_stock_qty(current_index, trade_dataset, entry_price, stop_loss, current_capital, L, L_hat, Λ, Λ_hat, signal_direction)
    if qty == 0:
        return None
    
    return FracDiffPosition(
        quote_name,
        current_index,
        entry_price,
        Λ,
        Λ_hat,
        signal_direction,
        qty,
        take_profit,
        stop_loss,
        [],
        entry_date = trade_dataset.index[current_index],
    )
