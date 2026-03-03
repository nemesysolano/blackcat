import json
import sys
import os

from sqlalchemy import create_engine
from qf.dbsync import read_quote_names, db_config
import pandas as pd

from qf.indicators import fractional_price_acceleration
from qf.nn import create_local_datasets
from qf.nn.models import fractional_order
from qf.trade import get_model_stats
from qf.trade import get_stats
from qf.trade.fracdiff import trade_fracdiff
import tensorflow as tf

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

def fractional_orders(Λ, L):
    orders = []
    
    for i in range(len(Λ)):
        t = L.index[i]
        orders.append(fractional_order(Λ[i], L.loc[t]))
    return tuple(orders)    
    

def create_trade_dataset(connection, quote_name, lookback_periods, model_name):
    indicator_data = (dataset, feature_names, target) = fractional_price_acceleration(connection, quote_name, lookback_periods)
    _, _, X_test, _, _, _, _ = create_local_datasets(indicator_data)    
    L = X_test[feature_names]
    Λ_hat = predictor(quote_name, model_name, X_test[feature_names])
    S = fractional_orders(Λ_hat, L)

    X_test = X_test.assign(
        OPEN = dataset.loc[X_test.index, 'OPEN'],
        LOW  = dataset.loc[X_test.index, 'LOW'],
        HIGH = dataset.loc[X_test.index, 'HIGH'],
        CLOSE= dataset.loc[X_test.index, 'CLOSE'],
        Λ_hat = Λ_hat,
        Λ = dataset.loc[X_test.index, 'Λ'],
        S = S       
    )
    print(X_test)
    return X_test, feature_names, target

def write_results(output_file, details_file, stats, transactions):
    mode = 'a' if os.path.exists(output_file) else 'w'
    with open(output_file, mode) as f:
        if mode == 'w':
            print(
                "Ticker, Initial Capital, Final Capital, Total Return (%), Max Drawdown (%), Volatility (per step), Sharpe Ratio, Number of Steps, Peak Equity, Final Drawdown (%), Long Trades, Short Trades, Winner Longs, Winner Shorts, Loser Longs, Loser Shorts",
                file=f
            )

        print(
            f"{stats['Ticker']}, {stats['Initial Capital']:.2f}, {stats['Final Capital']:.2f}, {stats['Total Return (%)']:.2f}, {stats['Max Drawdown (%)']:.2f}, {stats['Volatility (per step)']:.2f}, {stats['Sharpe Ratio']:.2f}, {stats['Number of Steps']}, {stats['Peak Equity']:.2f}, {stats['Final Drawdown (%)']:.2f}, {stats['Long Trades']}, {stats['Short Trades']}, {stats['Winner Longs']}, {stats['Winner Shorts']}, {stats['Loser Longs']}, {stats['Loser Shorts']}",
            file=f
        ) 

        with open(details_file, 'w') as f:
            transaction_list = []
            for transaction in transactions:
                transaction = {
                    "Entry Index": transaction.entry_index,
                    "Entry Date": transaction.entry_date.isoformat(),                    
                    "Exit Date": transaction.exit_date.isoformat(),
                    "Entry Price": float(transaction.entry_price),
                    "Side": int(transaction.side),
                    "Quantity": int(transaction.quantity),
                    "Take Profit": float(transaction.take_profit),
                    "Stop Loss": float(transaction.stop_loss),
                    "PL": float(transaction.pl),
                    "Exit Index": int(transaction.exit_index),
                    "Exit Reason": transaction.exit_reason,
                    "Λ": float(transaction.Λ),
                    "Λ_hat": float(transaction.Λ_hat),
                    "position_history": [{"index": s.index, "open_price": float(s.open_price), "high_price": float(s.high_price), "low_price": float(s.low_price), "close_price": float(s.close_price), "Λ": float(s.Λ), "Λ_hat": float(s.Λ_hat)} for s in transaction.state]
                }
                transaction_list.append(transaction)
            print(json.dumps(transaction_list, ensure_ascii = False), file=f)     

if __name__ == "__main__":
    model_name = "fractional-price-direction"
    try:
        if len(sys.argv) < 2:
            print("Usage: python trade_wavelets.py <quotes_file> [predictor]")
            sys.exit(1)

        quotes_file = sys.argv[1]
        _, sqlalchemy_url = db_config()
        quotes = read_quote_names(quotes_file)
    except Exception as e:
        print(f"Error reading quotes: {e}")
        quotes = []

    # Load Reports.
    fractional_price_direction_stats = get_model_stats(os.getcwd(), "report-fractional-price-direction.csv")
    lookback_periods = 14
    engine = create_engine(sqlalchemy_url)

    with engine.connect() as connection:
        for quote_name in quotes:
            quote_stats = get_stats(fractional_price_direction_stats, quote_name)
            tradable = quote_stats['tradable']
            
            if tradable:
                print(f"Backtesting {quote_name}")
                output_file = os.path.join(os.getcwd(), "test-results", f"report-backtest-fracdiff.csv")
                details_file = os.path.join(os.getcwd(), "test-results", f"report-{quote_name}-fracdiff-transactions.json")            
                if os.path.exists(details_file):
                    continue                    

                trade_dataset, feature_names, target_name = create_trade_dataset(connection, quote_name, lookback_periods, model_name)                                                
                stats, transactions = trade_fracdiff(quote_name, trade_dataset, lookback_periods, feature_names, target_name, f"{target_name}_hat")
                write_results(output_file, details_file, stats, transactions)
        connection.close()
    engine.dispose()
    