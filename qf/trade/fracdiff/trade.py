from qf.nativemath import get_fractional_integral as fractional_integral, get_fractional_integral_weights as fractional_integral_weights, get_fractional_signal as calculate_signal
from qf.nn.models.calculus import fractional_orders    
from qf.trade import create_backtest_stats
import pandas as pd
from qf.trade.fracdiff.sizing import create_position, update_position
from qf.trade.fracdiff.stats import FracDiffState
from qf.nn import create_local_datasets
import tensorflow as tf
import os
import numpy as np

def trade_fracdiff(quote_name, trade_dataset, lookback_periods, feature_names, target_name, estimation_name, max_leverage_allowed, direction_bias, platform_commission, initial_capital):  
    current_capital = initial_capital
    active_position = None
    transactions = []    
    long_trades, short_trades = 0, 0
    winner_longs, winner_shorts = 0, 0
    loser_longs, loser_shorts, cost= 0, 0, 0
    min_leverage, max_leverage = 0, 0
    equity_curve = [initial_capital]    
    gap_hit = False
    index = trade_dataset.index

    def add_transaction(t):
        nonlocal winner_longs, winner_shorts, loser_longs, loser_shorts, current_capital, equity_curve, cost, min_leverage, max_leverage
        transactions.append(t)  
        min_leverage = transaction.leverage if min_leverage == 0 or transaction.leverage < min_leverage else min_leverage
        max_leverage = transaction.leverage if max_leverage == 0 or transaction.leverage > max_leverage else max_leverage
        winner_longs = winner_longs + 1 if t.side == 1 and t.exit_reason == 1 else winner_longs
        loser_longs = loser_longs + 1 if t.side == 1 and t.exit_reason == -1 else loser_longs
        winner_shorts = winner_shorts + 1 if t.side == -1 and t.exit_reason == 1 else winner_shorts
        loser_shorts = loser_shorts + 1 if t.side == -1 and t.exit_reason == -1 else loser_shorts
        current_capital += t.pl
        cost += t.cost
        equity_curve.append(current_capital)        
        
    # Main simulation loop
    for current_index in range(lookback_periods, len(trade_dataset)-1):       
        t0 = index[current_index - 1]        
        t = index[current_index]       
        f = trade_dataset.loc[t, 'f']
        f0 = trade_dataset.loc[t0, 'f']
        window_f = trade_dataset['f'].iloc[current_index - lookback_periods : current_index]
        f_mean = window_f.mean()
        f_stdev = window_f.std()
        order = trade_dataset.loc[t, 'S']

        Lambda = trade_dataset.loc[t, target_name] # Current acceleration
        Lambda_hat  = trade_dataset.loc[t, estimation_name] # Predicted acceleration                

        L = trade_dataset.loc[t0, 'L']
        L_hat = trade_dataset.loc[t, 'L']

        signal = calculate_signal(L, L_hat, Lambda_hat, Lambda, f0, f, f_mean, f_stdev, order)            
        active_position, transaction, gap_hit = update_position(current_index, trade_dataset, L, Lambda_hat, Lambda, active_position, f, f_mean, f_stdev)

        if transaction is not None:     
            add_transaction(transaction)

        if active_position is None and not gap_hit:
            active_position = create_position(quote_name, current_index, signal, trade_dataset, f, f_mean, f_stdev, L, L_hat, Lambda_hat, Lambda, current_capital, max_leverage_allowed, direction_bias, platform_commission, order)
            long_trades = long_trades + 1 if active_position and active_position.side == 1 else long_trades
            short_trades = short_trades + 1 if active_position and active_position.side == -1 else short_trades
        if active_position is not None:
            active_position.state.append(FracDiffState(
                current_index,
                open_price = float(trade_dataset.loc[t, 'OPEN']),
                high_price = float(trade_dataset.loc[t, 'HIGH']),
                low_price = float(trade_dataset.loc[t, 'LOW']),
                close_price = float(trade_dataset.loc[t, 'CLOSE']),                
                Lambda = Lambda_hat,
                Lambda_hat = Lambda,
                L = L,
                L_hat = L_hat
            ))
    
    return create_backtest_stats(
        quote_name, equity_curve, long_trades, short_trades, 
        winner_longs, winner_shorts, loser_longs, loser_shorts, transactions, cost,
        min_leverage, max_leverage
    )

def predictor(quote_name, model_name, X_input):
    checkpoint_filepath = os.path.join(os.getcwd(), 'models', f'{quote_name}-{model_name}.keras')    
    try:
        if not os.path.exists(checkpoint_filepath):
            print(f"Warning: Model {checkpoint_filepath} not found.")
            return None

        model = tf.keras.models.load_model(checkpoint_filepath)    
    except Exception as e:        
        print(f"Error loading model: {e}")
        return None
    
    predictions = model.predict(X_input, verbose=0)
    return predictions


def enhance_inputs(X, dataset, model_name, quote_name, feature_names):
    features = X[feature_names]
    Lambda_hat = predictor(quote_name, model_name, features)
    S = fractional_orders(Lambda_hat, features)

    X = X.assign(
        OPEN = dataset.loc[X.index, 'open_price'],
        LOW  = dataset.loc[X.index, 'low_price'],
        HIGH = dataset.loc[X.index, 'high_price'],
        CLOSE= dataset.loc[X.index, 'close_price'],
        L = dataset.loc[X.index, 'L'],
        Lambda = dataset.loc[X.index, 'Lambda'],
        Lambda_hat = Lambda_hat,        
        f = dataset.loc[X.index, 'f'],
        S = np.nan_to_num(S, nan=1.0)
    )
    
    return X

def create_trade_dataset(connection, redis_connection, quote_name, lookback_periods, model_name, indicator):
    indicator_data = (dataset, feature_names, target) = indicator(connection, redis_connection, quote_name, lookback_periods)
    _, _, X_test, _, _, _, _ = create_local_datasets(indicator_data)    
    X_test = enhance_inputs(X_test, dataset, model_name, quote_name, feature_names)
   
    return None, X_test, feature_names, target
