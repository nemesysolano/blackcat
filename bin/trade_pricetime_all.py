import numpy as np
import json
import sys
import os
from sqlalchemy import create_engine
import traceback
from qf.dbsync import read_quote_names, db_config
import pandas as pd
from qf.indicators import log_acceleration_direction
from qf.trade.fracdiff import trade_fracdiff 
import redis
from qf.trade.fracdiff.trade import create_trade_generic_dataset

def write_results(output_file, details_file, stats, transactions):
    mode = 'a' if os.path.exists(output_file) else 'w'
    with open(output_file, mode) as f:
        if mode == 'w':
            print(
                "Ticker,Initial Capital,Final Capital,Cost,Net Capital,Total Return (%),Min Leverage,Max Leverage,Max Drawdown (%),Volatility (per step),Sharpe Ratio,Number of Steps,Peak Equity,Final Drawdown (%),Long Trades,Short Trades,Winner Longs,Winner Shorts,Loser Longs,Loser Shorts",
                file=f
            )

        print(
            f"{stats['Ticker']}, {stats['Initial Capital']:.2f}, {stats['Final Capital']:.2f}, {stats['Cost']:.2f}, {stats['Net capital']:.2f}, {stats['Total Return (%)']:.2f},{stats['Min Leverage']:.2f},{stats['Max Leverage']:.2f}, {stats['Max Drawdown (%)']:.2f}, {stats['Volatility (per step)']:.2f}, {stats['Sharpe Ratio']:.2f}, {stats['Number of Steps']}, {stats['Peak Equity']:.2f}, {stats['Final Drawdown (%)']:.2f}, {stats['Long Trades']}, {stats['Short Trades']}, {stats['Winner Longs']}, {stats['Winner Shorts']}, {stats['Loser Longs']}, {stats['Loser Shorts']}",
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
                    "Leverage": transaction.leverage,
                    "Side": int(transaction.side),
                    "Quantity": int(transaction.quantity),
                    "Take Profit": float(transaction.take_profit),
                    "Stop Loss": float(transaction.stop_loss),
                    "PL": float(transaction.pl),
                    "Signal": transaction.signal,
                    "Exit Index": int(transaction.exit_index),
                    "Exit Reason": transaction.exit_reason,
                    "Lambda": float(transaction.Lambda),
                    "Lambda_hat": float(transaction.Lambda_hat),
                    "L": float(transaction.L),
                    "L_hat": float(transaction.L_hat),
                    "Stallness Reason": transaction.stallness_reason,
                    "position_history": [{"index": s.index, "open_price": float(s.open_price), "high_price": float(s.high_price), "low_price": float(s.low_price), "close_price": float(s.close_price), "Lambda": float(s.Lambda), "Lambda_hat": float(s.Lambda_hat), "L": float(s.L), "L_hat": float(s.L_hat)} for s in transaction.state],
                    "trailing": [{"datetime": t.datetime.isoformat(), "take_profit": float(t.take_profit), "stop_loss": float(t.stop_loss)} for t in transaction.trailing]
                }
                transaction_list.append(transaction)
            print(json.dumps(transaction_list, ensure_ascii = False), file=f)     

def main(test_results_folder, model_name, max_leverage, platform_commission, initial_capital, lookback_periods):
    try:
        if len(sys.argv) < 2:
            print("Usage: python trade_fracdiff.py <quotes_file> [predictor]")
            sys.exit(1)

        quotes_file = sys.argv[1]
        _, sqlalchemy_url, redis_host, redis_port, redis_database = db_config()
        quotes = read_quote_names(quotes_file)
    except Exception as e:
        print(f"Error reading quotes: {e}")
        quotes = []

    result_name = f"backtest-{model_name}"    
    engine = create_engine(sqlalchemy_url)
    redis_connection = redis.Redis(host=redis_host, port=redis_port, db=redis_database, decode_responses=True)

    with engine.connect() as connection:        
        for quote_name in quotes:
            try:
                direction_bias = 1# np.sign(train_stats.loc[quote_name, 'Match %'] - train_stats.loc[quote_name, 'Different %'])
                print(f"Backtesting {quote_name} with {model_name} model.")
                output_file = os.path.join(os.getcwd(), test_results_folder, f"{result_name}.csv")
                details_file = os.path.join(os.getcwd(), test_results_folder, f"{result_name}-{quote_name}-transactions.json")            
                
                if os.path.exists(details_file):
                    continue                    

                _, trade_dataset, feature_names, target_name = create_trade_generic_dataset(connection, redis_connection, quote_name, lookback_periods, model_name, log_acceleration_direction)                  
                stats, transactions, _ = trade_fracdiff(quote_name, trade_dataset, lookback_periods, feature_names, target_name, f"{target_name}_hat", max_leverage, direction_bias, platform_commission, initial_capital)                
                write_results(output_file, details_file, stats, transactions)
            except Exception as cause:
                print(f"Error backtesting {quote_name}: {cause}")
                traceback.print_exc()
                
                continue


        connection.close()
    engine.dispose()

if __name__ == "__main__": 
    test_results_folder = os.path.join(os.getcwd(), "test-results")
    max_leverage = 10
    initial_capital = 50_000
    platform_commission = 6
    lookback_periods = 20
    main(test_results_folder, "price-time-wavelet-direction",  max_leverage, platform_commission, initial_capital, lookback_periods)

