
from qf.nn.models import fractional_integral_weights
from qf.nn.models import fractional_integral
from qf.trade import create_backtest_stats
from qf.trade.fracdiff.signal import STRONG_BULLISH, STRONG_BEARISH, MEAN_REVERSION_LONG, MEAN_REVERSION_SHORT, STALL
import numpy as np
import pandas as pd

from qf.trade.fracdiff.sizing import create_position, update_position
from qf.trade.fracdiff.stats import FracDiffState


def calculate_signal(L, L_hat, Λ, Λ_hat):
    # Sum of signs: 4 = All positive, -4 = All negative
    confluence = np.sign(Λ) + np.sign(Λ_hat) + np.sign(L) + np.sign(L_hat)
    
    if confluence == 4:
        return STRONG_BULLISH
    elif confluence == -4:
        return STRONG_BEARISH
    
    # Mean Reversion: Momentum is negative, but force is shifting positive
    elif (np.sign(L) + np.sign(L_hat) == -2) and (np.sign(Λ) + np.sign(Λ_hat) == 2):
        return MEAN_REVERSION_LONG
        
    # Mean Reversion: Momentum is positive, but force is shifting negative
    elif (np.sign(L) + np.sign(L_hat) == 2) and (np.sign(Λ) + np.sign(Λ_hat) == -2):
        return MEAN_REVERSION_SHORT
    
    return STALL
    
def trade_fracdiff(quote_name, trade_dataset, lookback_periods, feature_names, target_name, estimation_name, initial_capital = 10000.0):  
    current_capital = initial_capital
    active_position = None
    transactions = []    
    long_trades, short_trades = 0, 0
    winner_longs, winner_shorts = 0, 0
    loser_longs, loser_shorts = 0, 0
    equity_curve = []    

    for i in range(lookback_periods-1, len(trade_dataset)):
        t = trade_dataset.index[i]        
        if not pd.isna(trade_dataset.loc[t, 'S']):
            order = trade_dataset.loc[t, 'S']
            weights = fractional_integral_weights(order, lookback_periods)

            # 1. Extraction of Classical State
            L = trade_dataset.loc[t, feature_names[-1]] # Current log-return
            Λ = trade_dataset.loc[t, target_name]       # Actual acceleration
            
            # 2. Refined Extraction of Fractional State
            # We take the actual historical accelerations for N-1 periods
            historical_Λ = trade_dataset.loc[:t, target_name].iloc[-lookback_periods:].values
            
            # We replace the last (current) value with the model's prediction
            Λ_hat_t = trade_dataset.loc[t, estimation_name]
            
            # Create the hybrid vector: [Actual_{t-N}, Actual_{t-1}, Predicted_{t}]
            Λ_hybrid = historical_Λ.copy()
            Λ_hybrid[-1] = Λ_hat_t 
            
            # 3. Calculate Integrated Momentum (L_hat)
            L_hat = fractional_integral(weights, Λ_hybrid)

            # 4. Generate Signal based on Resonance Table
            signal = calculate_signal(L, L_hat, Λ, Λ_hat_t)            
            active_position, transaction = update_position(i, signal, trade_dataset, L, L_hat, Λ, Λ_hat_t, active_position)

            if transaction is not None:
                transactions.append(transaction)
                winner_longs = winner_longs + 1 if transaction.side == 1 and transaction.exit_reason == 1 else winner_longs
                loser_longs = loser_longs + 1 if transaction.side == 1 and transaction.exit_reason == -1 else loser_longs
                winner_shorts = winner_shorts + 1 if transaction.side == -1 and transaction.exit_reason == 1 else winner_shorts
                loser_shorts = loser_shorts + 1 if transaction.side == -1 and transaction.exit_reason == -1 else loser_shorts
                current_capital += transaction.pl
                equity_curve.append(current_capital)

            if active_position is None:
                active_position = create_position(quote_name, i, signal, trade_dataset, L, L_hat, Λ, Λ_hat_t, current_capital)
                long_trades = long_trades + 1 if active_position and active_position.side == 1 else long_trades
                short_trades = short_trades + 1 if active_position and active_position.side == -1 else short_trades
            
            if active_position is not None:
                active_position.state.append(FracDiffState(
                    i,
                    open_price = float(trade_dataset.loc[t, 'OPEN']),
                    high_price = float(trade_dataset.loc[t, 'HIGH']),
                    low_price = float(trade_dataset.loc[t, 'LOW']),
                    close_price = float(trade_dataset.loc[t, 'CLOSE']),                
                    Λ = Λ,
                    Λ_hat = Λ_hat_t,
                    L = L,
                    L_hat = L_hat
                ))
                
    if len(equity_curve) == 0:
        equity_curve.append(initial_capital)

    return create_backtest_stats(
        quote_name, equity_curve, long_trades, short_trades, 
        winner_longs, winner_shorts, loser_longs, loser_shorts, transactions
    )