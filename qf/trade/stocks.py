import numpy as np
from qf.trade import create_backtest_stats
from qf.trade import calculate_dynamic_qty, update_hybrid_exit, FLOOR, CEILING, calculate_levels
from qf.trade.model import Position
from qf.trade.model import Transaction
from qf.trade.model import State


def trade_stocks(quote_name, df, price_time_predictions, volume_time_predictions, force_predictions):    
    initial_capital = 10000.0
    current_capital = float(initial_capital)
    active_position = None
    transactions = []
    
    long_trades, short_trades = 0, 0
    winner_longs, winner_shorts = 0, 0
    loser_longs, loser_shorts = 0, 0
    equity_curve = []
    is_forex = quote_name.endswith("=X")
    is_jpy = "JPY" in quote_name
    current_floor = FLOOR * 0.9 if is_forex else FLOOR + 0.02
    current_ceiling = CEILING * 1.1 if is_forex else CEILING
    f_val0 = force_predictions[0][0]

    for i in range(len(df)):
        row = df.iloc[i]
        curr_dp = float(row['δP'])
        p_dir = int(np.sign(price_time_predictions[i][0]))
        v_dir = int(np.sign(volume_time_predictions[i][0]))
        f_val = force_predictions[i][0]
        close_series = row['CLOSE']
        H = row['H']
        V = row['V']        
        effective_dir =  p_dir if is_forex else (v_dir if v_dir != p_dir else 0)
        effective_V = V if not is_forex else 0        
        effective_H = H if not is_forex else 0        
        curr_close = float(close_series.to_numpy()[0] if hasattr(close_series, 'to_numpy') else close_series)
        
        # --- 1. EXIT LOGIC (Trailing Stop Update) ---
        if active_position:
            steps_held = i - active_position.entry_index
            leeway = (0.5 - effective_H) * 0.15
            decay = max(0, (5 - steps_held) / 5)
            buffered_floor = current_floor - (leeway * decay)
            updated_pos, exit_price , exit_reason = update_hybrid_exit(
                active_position, 
                row, 
                curr_dp, 
                f_val,
                is_forex,
                buffered_floor
            )
            
            if not (exit_price is None):
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
        if (active_position is None) and \
            ((not is_forex) or (is_forex and current_floor < f_val < current_ceiling)) and \
            (f_val >= f_val0) : # The Force Gradient Filter (Momentum Confirmation)

            # --- DYNAMIC FLOOR/CEILING CALIBRATION ---
            # alpha, beta, gamma are sensitivity coefficients
            h_shift = (0.5 - H) * 0.10     # Stricter if H < 0.5 (noise)
            v_shift = curr_dp * 2.0            # Stricter if Volatility is high
            e_shift = V * 0.05            # Relaxed if Energy is high
            
            # Calculate base dynamic bounds
            dynamic_floor_base = FLOOR + h_shift + v_shift - e_shift
            dynamic_ceiling_base = CEILING + h_shift + v_shift - e_shift

            # Preserve your original Forex/Stock logic on top of the dynamic base
            if is_forex:
                current_floor = max(0.08, dynamic_floor_base * 0.9)
                current_ceiling = min(0.30, dynamic_ceiling_base * 1.1)
            else:
                current_floor = max(0.10, dynamic_floor_base + 0.02)
                current_ceiling = min(0.35, dynamic_ceiling_base)

            confidence = (f_val - current_floor) / (current_ceiling - current_floor)
            if effective_dir != 0:

                # 1. Calculate Levels FIRST to get the Stop Loss (Volatility metric)
                tp_base, sl = calculate_levels(curr_close, effective_dir, curr_dp, effective_H, effective_V, is_forex, is_jpy)
                
                # 2. Calculate Quantity SECOND (Using the compounding risk logic)
                dynamic_qty = calculate_dynamic_qty(
                    confidence,
                    effective_H, 
                    effective_V,  
                    current_capital, 
                    curr_close,
                    sl,               # Pass the calculated stop loss
                    is_forex
                )
                
                # Guard clause
                if dynamic_qty > 0:
                    # 3. Finalize TP based on conviction
                    tp_dist = abs(tp_base - curr_close) * (1.0 + 0.5 * confidence)

                    if tp_dist > curr_close * curr_dp and effective_V <= 0.10: # if dist_to_SL < (close_price * dP): skip_trade(), if V > 0.10: skip_trade()    
                        final_tp = curr_close + (tp_dist * effective_dir)

                        active_position = Position(
                            ticker = quote_name,
                            entry_index = i,
                            entry_price =curr_close,
                            entry_force = f_val,
                            side = effective_dir,
                            quantity = dynamic_qty,
                            take_profit = float(final_tp),
                            stop_loss = float(sl),
                            state = []
                        )

                        commission = dynamic_qty * 0.005
                        current_capital -= commission   
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