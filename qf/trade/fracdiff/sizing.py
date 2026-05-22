import pandas as pd
from qf.trade import F_STD_K_FACTOR, SIGNAL_LABELS, STRONG_BULLISH, STRONG_BEARISH, MEAN_REVERSION_LONG, MEAN_REVERSION_SHORT
from qf.trade.fracdiff.stats import FracDiffPosition, FracDiffTransaction
from qf.trade import MAX_RISK_PER_TRADE
from qf.trade.fracdiff.stats.FracTrailingLimit import FracTrailingLimit
from qf.nativemath import get_levels as _calculate_levels, get_fractional_qty as calculate_stock_qty, get_fractional_physics_close as fractional_physics_close, get_fractional_update_levels as fractional_update_levels


def calculate_levels(current_index, signal, trade_dataset, f, f_mean, f_stdev, L, L_hat, Lambda, Lambda_hat, direction_bias, order, energy_signal, thrust_signal):
    t = trade_dataset.index[current_index]
    row = trade_dataset.loc[t]    
    current_price = row['CLOSE']
    high_price = row['HIGH']
    low_price = row['LOW']

    return _calculate_levels(signal, L, L_hat, Lambda, Lambda_hat, direction_bias, f, f_mean, f_stdev, current_price, low_price, high_price, order, energy_signal, thrust_signal)    

def try_early_close(current_index, position, current_open, current_high, current_low, exit_date):
    """
    Evaluates if the position hit its Stop Loss or Take Profit during the current day.
    Checks opening gaps first, then intraday wicks, generating a fully costed transaction.
    """
    is_gap = False
    exit_price = 0.0
    exit_reason = 0
    
    if position.side == 1: # LONG
        if current_open < position.stop_loss:
            exit_price, exit_reason, is_gap = current_open, -1, True
        elif current_open > position.take_profit:
            exit_price, exit_reason, is_gap = current_open, 1, True
        elif current_low <= position.stop_loss:
            exit_price, exit_reason = position.stop_loss, -1
        elif current_high >= position.take_profit:
            exit_price, exit_reason = position.take_profit, 1
            
    elif position.side == -1: # SHORT
        if current_open > position.stop_loss:
            exit_price, exit_reason, is_gap = current_open, -1, True
        elif current_open < position.take_profit:
            exit_price, exit_reason, is_gap = current_open, 1, True
        elif current_high >= position.stop_loss:
            exit_price, exit_reason = position.stop_loss, -1
        elif current_low <= position.take_profit:
            exit_price, exit_reason = position.take_profit, 1

    # If an exit condition was met
    if exit_price != 0.0:
        # Calculate gross PL (Commission & Swaps are handled by from_position)
        qty = getattr(position, 'quantity', getattr(position, 'qty', 0))
        profit_loss = (exit_price - position.entry_price) * qty * position.side
        
        # Label the transaction for your stats
        reason_label = "Take Profit" if exit_reason == 1 else "Stop Loss"
        if is_gap:
            reason_label += " (Gap)"
            
        # Delegate to the factory method to calculate holding_days, swaps, and costs
        transaction = FracDiffTransaction.from_position(
            position=position,
            current_index=current_index,
            exit_price=exit_price,
            exit_reason=exit_reason,
            profit_loss=profit_loss,
            exit_date=exit_date,
            stallness_reason=reason_label
        )
        return None, transaction, is_gap
        
    return position, None, False

def try_update_levels(position, low_price, high_price, L, Lambda, exit_date):
    new_sl, new_tp = fractional_update_levels(position.side, position.stop_loss, position.take_profit, position.entry_price, low_price, high_price, L, Lambda)    
    if new_sl != 0 and new_tp != 0:
        position = position._replace(stop_loss=new_sl, take_profit=round(new_tp, 4), trailing=position.trailing + [FracTrailingLimit(exit_date.to_pydatetime(), round(new_tp, 4), round(new_sl, 4))])
    return position


def try_eos_close(position, current_index, current_price, exit_date):   
    profit_loss = (current_price - position.entry_price) * position.quantity * position.side
    exit_reason = 1 if profit_loss > 0 else -1
    transaction = FracDiffTransaction.from_position(position, current_index, current_price, exit_reason, profit_loss, exit_date, "End of Stream")
    return None, transaction

def try_physics_close(current_index, position, eval_price, exec_price, exit_date, Lambda, Lambda_hat, f, f_mean, f_std):
    # 1. EVALUATION: Feed Day T's Close to the C++ Engine
    exit_reason, _cpp_pl, stallness_reason = fractional_physics_close(
        current_index, position.entry_index, position.entry_price, position.quantity, position.side, 
        eval_price, Lambda, Lambda_hat, f, f_mean, f_std
    )
    profit_loss = (exec_price - position.entry_price) * position.quantity * position.side

    if exit_reason != 0:
        stalness_label = "Physics Flip" if stallness_reason == 2 else "3-Sigma Volatility Ejection"
        
        # 2. EXECUTION: Calculate real PnL using Day T+1's Open
        qty = getattr(position, 'quantity', getattr(position, 'qty', 0))
        real_pl = (exec_price - position.entry_price) * qty * position.side
        
        # The from_position factory handles the swap and spread costs
        transaction = FracDiffTransaction.from_position(
            position=position, 
            current_index=current_index, 
            exit_price=exec_price, 
            exit_reason=exit_reason, 
            profit_loss=real_pl, 
            exit_date=exit_date, 
            stallness_reason=stalness_label
        )
        return None, transaction
    
    return position, None

def update_position(current_index, trade_dataset, L, Lambda, Lambda_hat, position, f, f_mean, f_std):        
    if position is None: 
        return None, None, False
        
    transaction = None
    exit_date = trade_dataset.index[current_index] # Day T
    
    current_open = trade_dataset.loc[exit_date, 'OPEN']
    current_high = trade_dataset.loc[exit_date, 'HIGH']
    current_low = trade_dataset.loc[exit_date, 'LOW']
    current_price = trade_dataset.loc[exit_date, 'CLOSE']
    
    # 1. Intraday Reality: Did it hit SL/TP during Day T?
    # This evaluates overnight gaps and intraday wicks using the updated try_early_close
    position, transaction, is_gap = try_early_close(
        current_index, position, current_open, current_high, current_low, exit_date
    )
    
    # 2. EOD Physics: Calculated at Close T, Executed at Open T+1
    if transaction is None and not (pd.isna(f_mean) or pd.isna(f_std)):
        if current_index + 1 < len(trade_dataset):
            next_date = trade_dataset.index[current_index + 1]
            next_open_price = trade_dataset.iloc[current_index + 1]['OPEN']
            
            ## Pass BOTH current_price (eval) and next_open_price (exec)
            position, transaction = try_physics_close(
                current_index, position, current_price, next_open_price, next_date, 
                Lambda, Lambda_hat, f, f_mean, f_std
            )

    # 3. Dynamic Updates: Position survived Day T, ratcheting limits based on geometry
    if transaction is None:
        position = try_update_levels(
            position, current_low, current_high, L, Lambda, exit_date
        )

    # 4. Force Close at End of Series
    if transaction is None and current_index == len(trade_dataset) - 2:
        position, transaction = try_eos_close(
            position, current_index, current_price, exit_date
        )        

    return position, transaction, is_gap

def create_position(quote_name, current_index, signal, trade_dataset, f, f_mean, f_stdev, L, L_hat, Lambda, Lambda_hat, current_capital, max_leverage_allowed, direction_bias, platform_commission, order, energy_signal, thrust_signal):
    t1 = trade_dataset.index[current_index + 1]
    take_profit, stop_loss, signal_direction = calculate_levels(current_index, signal, trade_dataset, f, f_mean, f_stdev, L, L_hat, Lambda, Lambda_hat, direction_bias, order, energy_signal, thrust_signal)
    entry_price = trade_dataset.loc[t1, 'OPEN']

    # qty and actual_leverage used
    qty, actual_leverage = calculate_stock_qty(entry_price, stop_loss, current_capital, L, L_hat, Lambda, Lambda_hat, max_leverage_allowed, platform_commission, order)
    if qty == 0: return None
    
    return FracDiffPosition(
        quote_name, current_index, entry_price, actual_leverage, Lambda, Lambda_hat, L, L_hat, signal_direction, qty,
        take_profit, stop_loss,
        SIGNAL_LABELS[signal], [], trade_dataset.index[current_index], [FracTrailingLimit(trade_dataset.index[current_index].to_pydatetime(), take_profit, stop_loss)]
    )