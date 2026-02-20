import numpy as np
from qf.trade import create_backtest_stats
from qf.trade import calculate_dynamic_qty, update_position, FLOOR, CEILING, calculate_levels
from qf.trade.model import Position
from qf.trade.model import Transaction
from qf.trade.model import State

def trade_forex(quote_name, df, price_time_predictions, volume_time_predictions, force_predictions):
    """
    Specialized trading logic for major Forex pairs (excluding JPY crosses if handled separately).
    Focuses on mean reversion and lower volatility thresholds.
    """
    initial_capital = 10000.0
    current_capital = float(initial_capital)
    active_position = None
    transactions = []
    
    long_trades, short_trades = 0, 0
    winner_longs, winner_shorts = 0, 0
    loser_longs, loser_shorts = 0, 0
    equity_curve = []
    
    # Forex Specific Constants
    # Tighter force bounds for Forex as it is less "explosive" than stocks
    FX_FLOOR = 0.10
    FX_CEILING = 0.25
    
    current_floor = FX_FLOOR
    current_ceiling = FX_CEILING
    f_val0 = force_predictions[0][0]

    for i in range(len(df)):
        row = df.iloc[i]
        curr_dp = float(row['δP'])
        p_dir = int(np.sign(price_time_predictions[i][0]))
        f_val = force_predictions[i][0]
        close_series = row['CLOSE']
        H = row['H']
        V = row['V']
        
        # Forex relies purely on Price-Time Wavelet Direction
        # Volume is often less reliable in decentralized FX markets
        effective_dir = p_dir 
        effective_V = V # Still track V for instability
        effective_H = H 
        
        curr_close = float(close_series.to_numpy()[0] if hasattr(close_series, 'to_numpy') else close_series)

        # --- 1. EXIT LOGIC (Standard + Time Decay) ---
        if active_position:
            # Forex Modification: Less aggressive "Early Stop"
            # Allow the trade to breathe for at least 3 hours (assuming 1h bars) or 12 steps
            steps_held = i - active_position.entry_index
            
            # Dynamic Floor for Exit is looser for Forex to prevent premature chop-out
            leeway = (0.5 - effective_H) * 0.20 # More leeway than stocks
            decay = max(0, (12 - steps_held) / 12) # longer decay (12 steps)
            buffered_floor = current_floor - (leeway * decay)

            updated_pos, exit_price , exit_reason = update_position(
                active_position, 
                row, 
                curr_dp, 
                f_val,
                True, # Always True for this function
                buffered_floor
            )
            
            if exit_price is not None:
                pl_total = float((exit_price - active_position.entry_price) * active_position.side * active_position.quantity)
                current_capital += pl_total

                if active_position.side == 1:
                    long_trades += 1
                    if pl_total > 0: winner_longs += 1
                    else: loser_longs += 1
                else:
                    short_trades += 1
                    if pl_total > 0: winner_shorts += 1
                    else: loser_shorts += 1

                transactions.append(Transaction.from_position(active_position, pl_total, i, exit_price, exit_reason))
                active_position = None
            else:
                active_position = updated_pos

        # --- 2. ENTRY LOGIC ---
        elif (active_position is None) and (current_floor < f_val < current_ceiling):
             # Forex Modification: Structural Gate
             # Only enter if the move is structurally sound (Low Instability)
             # V threshold is strictly 0.05 for Forex to avoid "whipsaws"
             if effective_V < 0.05:
                
                # Force Gradient is OPTIONAL for Forex Mean Reversion 
                # (sometimes we fade the move, but here we stick to trend following for safety)
                if f_val >= f_val0:
                    
                    confidence = (f_val - current_floor) / (current_ceiling - current_floor)
                    
                    # 1. Levels (Wider Stops for Forex)
                    # We pass is_forex=True, is_jpy=False (standard pairs)
                    tp_base, sl = calculate_levels(curr_close, effective_dir, curr_dp, effective_H, effective_V, True, False)
                    
                    # 2. Quantity (Higher Leverage allowed)
                    # calculate_dynamic_qty handles the 5x leverage logic internally for is_forex=True
                    dynamic_qty = calculate_dynamic_qty(
                        confidence, effective_H, effective_V,  
                        current_capital, curr_close, sl, True
                    )
                    
                    if dynamic_qty > 0:
                        # 3. TP Adjustment
                        # Forex targets are often smaller relative to price (pip-hunting)
                        tp_dist = abs(tp_base - curr_close)
                        final_tp = curr_close + (tp_dist * effective_dir)

                        active_position = Position(
                            ticker = quote_name,
                            entry_index = i,
                            entry_price = curr_close,
                            entry_force = f_val,
                            side = effective_dir,
                            quantity = dynamic_qty,
                            take_profit = float(final_tp),
                            stop_loss = float(sl),
                            state = []
                        )
                        # No commission for Forex (usually spread-based, modeled in exit price if needed)
                        # current_capital -= commission 

        # --- 3. EQUITY TRACKING ---
        unrealized = 0.0
        if active_position:
            unrealized = float((curr_close - active_position.entry_price) * active_position.side * active_position.quantity)
            active_position.state.append(
                State(
                    index = i,
                    open_price = float(row['OPEN']),
                    high_price = float(row['HIGH']),
                    low_price = float(row['LOW']),
                    close_price = curr_close,
                    δP = curr_dp,
                    V = effective_V,
                    H = H,
                    previous_force= f_val0,
                    current_force=f_val
                )
            )

        equity_curve.append(float(current_capital + unrealized))
        f_val0 = f_val

    return create_backtest_stats(
        quote_name, equity_curve, long_trades, short_trades, 
        winner_longs, winner_shorts, loser_longs, loser_shorts, transactions
    )