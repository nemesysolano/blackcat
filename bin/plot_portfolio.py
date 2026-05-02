import sys
import os
import json
import pandas as pd
from sqlalchemy import create_engine
from qf.dbsync import db_config
import mplfinance as mpf
import matplotlib.pyplot as plt

def create_plot_index(test_results_dir, tickers, backtesting_results_base_file):
    transactions = dict()
    min_time_stamp = None
    max_time_stamp = None

    for ticker in tickers:
        transactions_file = os.path.join(test_results_dir, f"{backtesting_results_base_file}-{ticker}-transactions.json")
        with open(transactions_file, 'r') as f:
            ticker_transactions = json.load(f)
            if len(ticker_transactions) == 0:
                continue
            
            transactions[ticker] = ticker_transactions            
            first_exit = pd.to_datetime(ticker_transactions[0]['Exit Date'])
            last_exit = pd.to_datetime(ticker_transactions[-1]['Exit Date'])

            if min_time_stamp is None or first_exit < min_time_stamp:
                min_time_stamp = first_exit
            if max_time_stamp is None or last_exit > max_time_stamp:
                max_time_stamp = last_exit

    min_time_stamp = min_time_stamp if min_time_stamp else None
    max_time_stamp = max_time_stamp if max_time_stamp else None
    plot_index = tuple([time_stamp for time_stamp in pd.date_range(min_time_stamp, max_time_stamp).tolist() if time_stamp.weekday() < 5])
    return plot_index, transactions

def create_equity_dataframe(plot_index, tickers):
    equity_data_frame = pd.DataFrame(0.0, index=plot_index, columns=tickers).rename_axis('quote_timestamp')
    return equity_data_frame

def fill_equity_dataframe(equity_data_frame, transactions):
    # Step 1: Record daily P/L events (Same as original)
    for ticker, ticker_transactions in transactions.items():
        for transaction in ticker_transactions:
            exit_date = pd.to_datetime(transaction['Exit Date'])
            pl = transaction['PL']
            if exit_date in equity_data_frame.index:
                equity_data_frame.loc[exit_date, ticker] += pl

    # Step 2: Convert daily P/L events into a running Cumulative Equity curve per ticker.
    # This prevents the "zero-out" drop off by carrying held capital forward.
    for ticker in transactions.keys():
        equity_data_frame[ticker] = equity_data_frame[ticker].cumsum()

    # Step 3: Calculate Aggregate Metrics (Refactored to match cumulative data)
    # Total portfolio equity is now simply the sum of all individual cumulative equities
    running_total = equity_data_frame.sum(axis=1)
    
    # Portfolio Daily P/L is the difference between today's total and yesterday's total
    pl = running_total.diff().fillna(0) 

    equity_data_frame['P/L'] = pl
    equity_data_frame['Equity'] = running_total
    
    return equity_data_frame

if __name__ == "__main__":
    test_results_dir = os.path.join(os.getcwd(), "test-results")
    backtesting_results = sys.argv[1]
    backtesting_results_base_file = os.path.basename(backtesting_results).replace(".csv", "")
    equity_csv_file = os.path.join(test_results_dir, f"{backtesting_results_base_file}-equity.csv")
    portfolio_dataframe = pd.read_csv(backtesting_results, index_col='Ticker')
    tickers = tuple(portfolio_dataframe.index.to_list())
    plot_index, transactions = create_plot_index(test_results_dir, tickers, backtesting_results_base_file)
    equity_data_frame = fill_equity_dataframe(create_equity_dataframe(plot_index, tickers), transactions)
    equity_data_frame.to_csv(equity_csv_file)
   
    