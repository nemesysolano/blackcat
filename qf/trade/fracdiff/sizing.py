import math
import numpy as np
from qf.trade.fracdiff.signal import STALL, STRONG_BULLISH, STRONG_BEARISH, MEAN_REVERSION_LONG, MEAN_REVERSION_SHORT
from qf.trade.fracdiff.stats import FracDiffPosition, FracDiffTransaction
from qf.trade.fracdiff.utils import MAX_RISK_PER_TRADE, TRANSACTION_COMMISSION

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

def try_normal_close(current_index, position, low_price, high_price, exit_date, L, L_hat, Λ, Λ_hat_t):
    exit_reason = 0
    
    # 1. Calculate the Quantum Confidence Score
    # conviction: How strongly the current and predicted forces push in our direction
    conviction = np.tanh((Λ + Λ_hat_t) * position.side * 10)
    
    # exhaustion: If level (L) is high in our trade direction, we are hitting a barrier
    exhaustion = np.tanh(max(0, L * position.side) * 5)
    
    # Final score (-1 to 1): High force + Low exhaustion = High Score
    alpha_score = conviction - (0.3 * exhaustion)

    # 2. Dynamic Threshold Mapping
    # High alpha_score (Great trade) -> Higher thresholds (0.65 / 0.85) to allow "tunneling"
    # Low alpha_score (Stalling/Exhausted) -> Lower thresholds (0.35 / 0.65) to lock in profit early
    be_threshold = np.clip(0.5 + (0.15 * alpha_score), 0.35, 0.65)
    lock_threshold = np.clip(0.75 + (0.10 * alpha_score), 0.65, 0.85)

    # 3. Apply Dynamic Trailing Logic
    tp_distance = abs(position.take_profit - position.entry_price)
    current_best = high_price if position.side == 1 else low_price
    progress = abs(current_best - position.entry_price) / tp_distance
    
    # Lock-in Rule (Dynamic Lock Threshold)
    if progress >= lock_threshold:
        lock_in_price = position.entry_price + (position.side * tp_distance * 0.5)
        position = position._replace(stop_loss=max(position.stop_loss, lock_in_price) if position.side == 1 else \
                                     min(position.stop_loss, lock_in_price))
    
    # Break-even Rule (Dynamic BE Threshold)
    elif progress >= be_threshold:
        new_sl = max(position.stop_loss, position.entry_price) if position.side == 1 else \
                 min(position.stop_loss, position.entry_price)
        position = position._replace(stop_loss=new_sl)

    # Standard Price-based Exit Check
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

def try_stallness_close(active_position, current_index, current_price, signal, trade_dataset, L, L_hat, Λ, Λ_hat):
    """
    Evaluates if an actively winning or losing position is stalling, eroding, or turning toxic.
    Returns (None, Transaction) if closed.
    """
    if not active_position.state:
        return active_position, None
        
    time_in_trade = len(active_position.state)
    entry_price = active_position.entry_price
    side = active_position.side

    # 1. PnL Tracking
    current_pnl_pct = ((current_price - entry_price) * side) / entry_price
    history_prices = [s.high_price if side == 1 else s.low_price for s in active_position.state]
    peak_price = max(history_prices + [current_price]) if side == 1 else min(history_prices + [current_price])
    peak_pnl_pct = ((peak_price - entry_price) * side) / entry_price
    
    stallness_reason = None
    should_exit = False
    
    # =================================================================
    # TIER 1: ABSOLUTE SIGNAL FAILURE
    # =================================================================
    # A. Quantum Inversion (The Circuit Breaker)
    confluence_score = np.sign(Λ) + np.sign(Λ_hat) + np.sign(L) + np.sign(L_hat)
    if (side == 1 and confluence_score <= -2) or (side == -1 and confluence_score >= 2):
        stallness_reason = f"Quantum Inversion: score={confluence_score}."
        should_exit = True

    # =================================================================
    # TIER 2: CAPITAL PROTECTION ON WINNERS
    # =================================================================
    # B. The "Free Trade" Rule (Hard Breakeven)
    if not should_exit and peak_pnl_pct >= 0.01:
        if current_pnl_pct <= 0.001:
            stallness_reason = f"Hard Breakeven: peak_pnl_pct={peak_pnl_pct:.4f}, current_pnl_pct={current_pnl_pct:.4f}."
            should_exit = True

    # C. Tiered Profit Erosion (Dynamic Protection)
    if not should_exit and peak_pnl_pct >= 0.01:
        retracement = (peak_pnl_pct - current_pnl_pct) / peak_pnl_pct
        
        if peak_pnl_pct >= 0.03:      # > 3.0% gain: allow 35% give-back
            allowed_retracement = 0.35
        elif peak_pnl_pct >= 0.02:    # 2.0% - 3.0% gain: allow 50% give-back
            allowed_retracement = 0.50
        else:                         # 1.0% - 2.0% gain: allow 70% give-back
            allowed_retracement = 0.70
            
        if retracement > allowed_retracement and current_pnl_pct > -0.005: 
            stallness_reason = f"Profit Erosion: allowed={allowed_retracement}, retraced={retracement:.2f}."
            should_exit = True

    # =================================================================
    # TIER 3: CUTTING LOSERS EARLY (CTVA/LHX Fix)
    # =================================================================
    # D. Failed Tunneling (Replaces 5-day Toxic Stagnation)
    if not should_exit and time_in_trade >= 3 and current_pnl_pct < 0:
        if (side == 1 and Λ < 0) or (side == -1 and Λ > 0):
            stallness_reason = f"Failed Tunneling: days={time_in_trade}, side={side}, Λ={Λ:.4f}."
            should_exit = True
            
    # E. Loser Erosion (Deep red, dead momentum)
    if not should_exit and current_pnl_pct < 0:
        sl_distance_pct = abs(active_position.stop_loss - entry_price) / entry_price
        if sl_distance_pct > 0:
            loss_ratio = abs(current_pnl_pct) / sl_distance_pct
            # If we've eaten 50% of our stop loss risk and total momentum is flat
            if loss_ratio >= 0.5 and (abs(Λ) + abs(Λ_hat)) < 0.002:
                stallness_reason = f"Loser Erosion: loss_ratio={loss_ratio:.2f}, flat momentum."
                should_exit = True

    # =================================================================
    # TIER 4: FADING MOMENTUM & TIME STOPS
    # =================================================================
    # F. Quantum Divergence (The Early Warning)
    if not should_exit and current_pnl_pct > 0.002:
        if (side == 1 and Λ < 0 and Λ_hat < 0) or (side == -1 and Λ > 0 and Λ_hat > 0):
            stallness_reason = f"Quantum Divergence: Λ={Λ:.4f}, Λ_hat={Λ_hat:.4f}."
            should_exit = True

    # G. Flat Stagnation
    if not should_exit and time_in_trade >= 8 and current_pnl_pct < 0.004:
        stallness_reason = f"Flat Stagnation: time={time_in_trade}, pnl={current_pnl_pct:.4f}."
        should_exit = True

    # =================================================================
    # EXECUTION
    # =================================================================
    if should_exit:
        exit_date = trade_dataset.index[current_index]                
        profit_loss = (current_price - entry_price) * active_position.quantity * side
        exit_reason = 1 if profit_loss > 0 else -1
        
        transaction = FracDiffTransaction.from_position(
            active_position, current_index, current_price, exit_reason, 
            profit_loss, exit_date, stallness_reason
        )        
        return None, transaction

    return active_position, None


def update_position(current_index, signal, trade_dataset, L, L_hat, Λ, Λ_hat_t, position):        
    if position is None: 
        return None, None
    
    transaction = None
    exit_date = trade_dataset.index[current_index]

    # --- Price-Based Exits (Gaps) ---
    if current_index == position.entry_index and current_index < len(trade_dataset) - 1:
        next_date = trade_dataset.index[current_index + 1]
        next_open_price = trade_dataset.loc[next_date, 'OPEN']
        position, transaction = try_early_close(current_index, position, next_open_price, next_date)

    # --- Price-Based Exits (Intraday with Dynamic Thresholds) ---
    if transaction is None:
        low_price = trade_dataset.loc[exit_date, 'LOW']
        high_price = trade_dataset.loc[exit_date, 'HIGH']
        position, transaction = try_normal_close(current_index, position, low_price, high_price, exit_date, L, L_hat, Λ, Λ_hat_t)
    
    # -- Stallness based exits.
    if transaction is None:
        current_price = trade_dataset.loc[exit_date, 'CLOSE']
        position, transaction = try_stallness_close(position, current_index, current_price, signal, trade_dataset, L, L_hat, Λ, Λ_hat_t)
    return position, transaction


def calculate_stock_qty(current_index, trade_dataset, entry_price, stop_loss, current_capital, L, L_hat, Λ, Λ_hat, signal_direction):
    """
    Calculates quantity based on Risk-at-Risk (3% of capital) and 
    scales the conviction based on the Magnitude of the Resonance.
    """
    if (stop_loss == 0) or (entry_price == stop_loss):
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
    max_affordable = current_capital / entry_price # TODO: Introduce friction allowance later
    
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
        L,
        L_hat,
        signal_direction,
        qty,
        take_profit,
        stop_loss,
        [],
        entry_date = trade_dataset.index[current_index],
    )
