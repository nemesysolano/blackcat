import numpy as np
from qf.trade import create_backtest_stats
from qf.trade.model import Position
from qf.trade.model import Transaction
from qf.trade.model import State
from .sizing import calculate_stock_levels, update_stock_position, calculate_stock_dynamic_qty


OMEGA_MAX = 0.20
FLOOR =0.13
CEILING = 0.18


def short_delta_filter(δP, δf, H, V, effective_direction):
    return (effective_direction == -1 and 
            δf > 0.00025 and 
            δP > 0.0025 and 
            V < 0.1 and 
            H > 0.2) 

def long_delta_filter(δP, δf, H, V, effective_direction):
    return (effective_direction == 1 and 
            δf > 0.00025 and 
            δP > 0.0025 and 
            V < 0.1 and 
            H > 0.2) 

def trade_wavelets(quote_name, df, price_time_predictions, volume_time_predictions, force_predictions, initial_capital = 10000.0, stop_and_reverse = False):    
    current_capital = float(initial_capital)
    active_position = None
    transactions = []
    
    long_trades, short_trades = 0, 0
    winner_longs, winner_shorts = 0, 0
    loser_longs, loser_shorts = 0, 0
    equity_curve = []
    f_val0 = force_predictions[0][0]
    abort_reverse = 0

    for i in range(len(df)):
        t = df.index[i]
        row = df.iloc[i]
        next_row = df.iloc[i+1] if i < len(df) - 1 else None
        δP = float(row['δP'])
        p_dir = int(np.sign(price_time_predictions[i][0]))
        v_dir = int(np.sign(volume_time_predictions[i][0])) 
        f_val = force_predictions[i][0]
        close_series = row['CLOSE']
        effective_dir =   effective_dir = (v_dir if v_dir != p_dir else p_dir) * (-1 if abort_reverse != 0 else 1)
        abort_reverse = 0
        current_price = float(close_series.to_numpy()[0] if hasattr(close_series, 'to_numpy') else close_series)
        δf = (f_val - f_val0)/(f_val + f_val0 + 0.000009)

        # --- 1. EXIT LOGIC (Trailing Stop Update) ---
        if active_position:
            updated_pos, exit_price , exit_reason = update_stock_position(i, df, active_position, δf) # In this line, δf ranges in [-1,1]
            
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

                transactions.append(Transaction.from_position(active_position, pl_total, i, exit_price, exit_reason, t))
                active_position = None
            else:
                active_position = updated_pos
        if active_position is None and (\
                δf > 0 and \
                (
                    short_delta_filter(δP, δf, row['H'], row['V'], effective_dir) or \
                    long_delta_filter(δP, δf, row['H'], row['V'],effective_dir)
                )
            ) :

            take_profit, stop_loss = calculate_stock_levels(i, df, current_price, effective_dir, δf)   # In this line, δf ranges in (0,1]                   
            dynamic_qty = calculate_stock_dynamic_qty(i, df, current_capital, current_price, stop_loss,  δf)# In this line, δf ranges in (0,1]      
            
            if dynamic_qty > 0:
                
                # 3. Finalize TP based on conviction
                active_position = Position(
                    ticker = quote_name,
                    entry_index = i,
                    entry_date= t,
                    entry_price  =current_price,
                    δf = δf,
                    δP = δP,
                    side = effective_dir,
                    quantity = dynamic_qty,
                    take_profit = float(take_profit),
                    stop_loss = float(stop_loss),
                    V = row['V'],
                    H = row['H'],                    
                    state = []
                )

                commission = dynamic_qty * 0.005
                current_capital -= commission                   
            # --- 3. EQUITY TRACKING ---
            unrealized = 0.0
            if active_position:
                # Exits if next open breaks stop loss or take profit.
                
                if (active_position.entry_index == i and not (next_row is None)):
                    next_open = float(next_row['OPEN'])
                    abort_loss = 0                                        
                    if (active_position.side == 1 and next_open < active_position.stop_loss) or (active_position.side == -1 and next_open > active_position.stop_loss):
                        abort_loss = float((next_open - active_position.entry_price) * active_position.side * active_position.quantity)
                        current_capital += (abort_loss - commission)
                        loser_longs += (1 if active_position.side == 1 else 0)
                        loser_shorts += (1 if active_position.side == -1 else 0)
                        transactions.append(Transaction.from_position(active_position, abort_loss, i, next_open, -1, t))
                        abort_reverse =  (- effective_dir if stop_and_reverse else 0)
                        unrealized = abort_loss
                    elif (active_position.side == 1 and next_open > active_position.take_profit) or (active_position.side == -1 and next_open < active_position.take_profit):
                        abort_profit = float((next_open - active_position.entry_price) * active_position.side * active_position.quantity)
                        current_capital += (abort_profit - commission)
                        winner_longs += (1 if active_position.side == 1 else 0)
                        winner_shorts += (1 if active_position.side == -1 else 0)
                        transactions.append(Transaction.from_position(active_position, abort_profit, i, next_open, 1, t))
                        abort_reverse = (- effective_dir if stop_and_reverse else 0)
                        unrealized = abort_profit
                        
                    active_position = None if abort_reverse != 0 else active_position
                else:
                    unrealized = float((current_price - active_position.entry_price) * active_position.side * active_position.quantity)
                    active_position.state.append(
                        State(
                            index = i,
                            open_price = float(row['OPEN']),
                            high_price = float(row['HIGH']),
                            low_price = float(row['LOW']),
                            close_price = current_price,
                            δP = δP,
                            V = row['V'],
                            H = row['H'],
                            δf = δf
                        )
                    )

            equity_curve.append(float(current_capital + unrealized))
        f_val0 = f_val

    if len(equity_curve) == 0:
        equity_curve.append(initial_capital)

    return create_backtest_stats(
        quote_name, equity_curve, long_trades, short_trades, 
        winner_longs, winner_shorts, loser_longs, loser_shorts, transactions
    )