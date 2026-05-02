import math
import numpy as np
from qf.trade import F_STD_K_FACTOR, SIGNAL_LABELS, STRONG_BULLISH, STRONG_BEARISH, MEAN_REVERSION_LONG, MEAN_REVERSION_SHORT
from qf.trade.fracdiff.stats import FracDiffPosition, FracDiffTransaction
from qf.trade import MAX_RISK_PER_TRADE
from qf.trade.fracdiff.stats.FracTrailingLimit import FracTrailingLimit
from qf.nativemath import get_levels as _calculate_levels, get_fractional_qty as calculate_stock_qty, get_fractional_physics_close as fractional_physics_close, get_fractional_update_levels as fractional_update_levels


def calculate_levels(current_index, signal, trade_dataset, L, L_hat, Lambda, Lambda_hat, direction_bias, order):
    t = trade_dataset.index[current_index]
    row = trade_dataset.loc[t]    
    current_price = row['CLOSE']
    f = row['f']
    f_mean = row['f_mean']
    f_stdev = row['f_stdev']
    high_price = row['HIGH']
    low_price = row['LOW']

    return _calculate_levels(signal, L, L_hat, Lambda, Lambda_hat, direction_bias, f, f_mean, f_stdev, current_price, low_price, high_price, order)    

def try_early_close(current_index, position, next_open_price, exit_date):
    profit_loss = None
    if position.side == 1 and (next_open_price < position.stop_loss or next_open_price > position.take_profit):
        profit_loss = (next_open_price - position.entry_price) * position.quantity * position.side 
    elif position.side == -1 and (next_open_price > position.stop_loss or next_open_price < position.take_profit):
        profit_loss = (next_open_price - position.entry_price) * position.quantity * position.side

    if profit_loss is not None:            
        exit_reason = 1 if profit_loss > 0 else -1
        return None, FracDiffTransaction.from_position(position, current_index, next_open_price, exit_reason, profit_loss, exit_date)
    return position, None 

def try_update_levels(position, low_price, high_price, L, Lambda, exit_date):
    new_sl, new_tp = fractional_update_levels(position.side, position.stop_loss, position.take_profit, position.entry_price, low_price, high_price, L, Lambda)    
    if new_sl != 0 and new_tp != 0:
        position = position._replace(stop_loss=new_sl, take_profit=round(new_tp, 4), trailing=position.trailing + [FracTrailingLimit(exit_date.to_pydatetime(), round(new_tp, 4), round(new_sl, 4))])
    return position


def try_normal_close(current_index, position, low_price, high_price, exit_date):
    exit_reason = 0

    if position.side == 1:
        if low_price < position.stop_loss: exit_reason = -1
        elif high_price > position.take_profit: exit_reason = 1
    elif position.side == -1:
        if high_price > position.stop_loss: exit_reason = -1
        elif low_price < position.take_profit: exit_reason = 1

    if exit_reason != 0:
        actual_exit_price = position.stop_loss if exit_reason == -1 else position.take_profit
        profit_loss = (actual_exit_price - position.entry_price) * position.quantity * position.side
        reason_label = "Take Profit" if exit_reason == 1 else "Stop Loss"
        return None, FracDiffTransaction.from_position(position, current_index, actual_exit_price, exit_reason, profit_loss, exit_date, f"{reason_label}")
        
    return position, None

def try_eos_close(position, current_index, current_price, exit_date):   
    profit_loss = (current_price - position.entry_price) * position.quantity * position.side
    exit_reason = 1 if profit_loss > 0 else -1
    transaction = FracDiffTransaction.from_position(position, current_index, current_price, exit_reason, profit_loss, exit_date, "End of Stream")
    return None, transaction

def try_physics_close(current_index, position, current_price, exit_date, Lambda, Lambda_hat, f, f_mean, f_std):
    exit_reason, profit_loss, snallness_reason = fractional_physics_close(current_index, position.entry_index, position.entry_price, position.quantity, position.side, current_price, Lambda, Lambda_hat, f, f_mean, f_std)
    if exit_reason != 0:
        stalness_label = "Physics Flip" if snallness_reason == 2 else "3-Sigma Volatility Ejection"
        return None, FracDiffTransaction.from_position(position, current_index, current_price, exit_reason, profit_loss, exit_date, stalness_label)
    
    return position, None

def update_position(current_index, trade_dataset, L, Lambda, Lambda_hat, position, f, f_mean, f_std):        
    gap_hit = False
    if position is None: return None, None, gap_hit
    transaction = None
    exit_date = trade_dataset.index[current_index]
    current_price = trade_dataset.loc[exit_date, 'CLOSE']

    if current_index == position.entry_index + 1:
        today_open_price = trade_dataset.loc[exit_date, 'OPEN']        
        position, transaction = try_early_close(current_index, position, today_open_price, exit_date    )
        gap_hit = not (transaction is None)

    if transaction is None:
        position = try_update_levels(position, trade_dataset.loc[exit_date, 'LOW'], trade_dataset.loc[exit_date, 'HIGH'], L, Lambda, exit_date)

    if transaction is None:
        low_price = trade_dataset.loc[exit_date, 'LOW']
        high_price = trade_dataset.loc[exit_date, 'HIGH']
        position, transaction = try_normal_close(current_index, position, low_price, high_price, exit_date)
    
    if transaction is None:
        position, transaction = try_physics_close(current_index, position, current_price, exit_date, Lambda, Lambda_hat, f, f_mean, f_std)

    if transaction is None and current_index == len(trade_dataset) - 2:
       position, transaction = try_eos_close(position, current_index, current_price, exit_date)        

    return position, transaction, gap_hit

def create_position(quote_name, current_index, signal, trade_dataset, L, L_hat, Lambda, Lambda_hat, current_capital, max_leverage_allowed, direction_bias, platform_commission, order):
    t1 = trade_dataset.index[current_index + 1]
    take_profit, stop_loss, signal_direction = calculate_levels(current_index, signal, trade_dataset, L, L_hat, Lambda, Lambda_hat, direction_bias, order)
    entry_price = trade_dataset.loc[t1, 'OPEN']

    # qty and actual_leverage used
    qty, actual_leverage = calculate_stock_qty(entry_price, stop_loss, current_capital, L, L_hat, Lambda, Lambda_hat, max_leverage_allowed, platform_commission, order)
    if qty == 0: return None
    
    return FracDiffPosition(
        quote_name, current_index, entry_price, actual_leverage, Lambda, Lambda_hat, L, L_hat, signal_direction, qty,
        take_profit, stop_loss,
        SIGNAL_LABELS[signal], [], trade_dataset.index[current_index], [FracTrailingLimit(trade_dataset.index[current_index].to_pydatetime(), take_profit, stop_loss)]
    )