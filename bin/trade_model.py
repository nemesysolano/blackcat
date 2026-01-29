from bin.train_model import directional_mse
from qf.indicators import volumeprice, bardirection
from qf.indicators import ensemble
from qf.indicators import wavelets
from qf.trade.stats import create_backtest_stats
import sys
import os
from qf.dbsync import read_quote_names, db_config
from qf.nn.splitter import create_trade_datasets
from qf.trade import Position, Transaction
from qf.trade.stats import get_returns
import tensorflow as tf
import pandas as pd
import qf.trade as trade
from sqlalchemy import create_engine
import numpy as np
import traceback

def get_quotes(sqlalchemy_url, quote_name, X_test):
    engine = create_engine(sqlalchemy_url)
    sql_template = f"SELECT \"TIMESTAMP\", \"OPEN\", \"HIGH\", \"LOW\" FROM QUOTE WHERE  \"TICKER\" = '{quote_name}'"
    with engine.connect() as connection:
        df = pd.read_sql(sql_template, connection)
        connection.close()
        df.set_index('TIMESTAMP', inplace=True)    
        df = df.loc[X_test.index]
        df = df.sort_index()
        df = df.join(X_test)
        assert len(df) == len(X_test)

    engine.dispose()
    return df

def trade_quotes(sqlalchemy_url, quote_name, X_test, indicators, edge, contrarian, structural_check, scalping):
    # 1. Prepare price data and merge with model features/angles
    df = get_quotes(sqlalchemy_url, quote_name, X_test)
    n = len(df)
    
    # Initialize backtest state
    active_position = None
    transaction_log = []
    cash = 10000.0
    equity_curve = []
    
    # Counters for stats
    long_trades = 0
    short_trades = 0
    winner_longs = 0
    winner_shorts = 0
    loser_longs = 0
    loser_shorts = 0

    for i in range(n):
        row = df.iloc[i]
        current_price = row['CLOSE']
        
        # --- PHASE 1: CHECK FOR EXITS ---
        if active_position is not None:
            # Transaction.from_position requires 7 arguments
            transaction = Transaction.from_position(
                active_position, 
                i, 
                row['OPEN'], 
                row['LOW'], 
                row['HIGH'], 
                row['CLOSE'],
                row['ATR %'],
                scalping
            )
            
            if transaction is not None:
                # Trade Closed: Record results and update counters
                transaction_log.append(transaction)
                cash += transaction.pl
                
                if transaction.side == 1:
                    if transaction.pl > 0: winner_longs += 1
                    else: loser_longs += 1
                else:
                    if transaction.pl > 0: winner_shorts += 1
                    else: loser_shorts += 1
                
                active_position = None # Reset for next trade

        # --- PHASE 2: CHECK FOR ENTRIES ---
        # Only enter if we don't have an open position and aren't at the very last bar
        if active_position is None and i < n - 1:
            # Position.create implements the angle "Double Check"
            # Returns None if Momentum and Structure diverge
            active_position = Position.create(
                quote_name, 
                df, 
                i, 
                indicators[i], 
                edge, 
                current_price,
                contrarian,
                structural_check
            )
            
            if active_position is not None:
                if active_position.side == 1: long_trades += 1
                else: short_trades += 1

        # --- PHASE 3: TRACK EQUITY ---
        current_equity = cash
        if active_position is not None:
            # Unrealized P&L calculation
            if active_position.side == 1:
                current_equity += (current_price - active_position.entry_price) * active_position.quantity
            else:
                current_equity += (active_position.entry_price - current_price) * active_position.quantity
        
        equity_curve.append(current_equity)

    return (
        quote_name, 
        equity_curve, 
        long_trades, 
        short_trades, 
        winner_longs, 
        winner_shorts, 
        loser_longs, 
        loser_shorts, 
        transaction_log
    )

def generate_reports(results, indicator_name, quote_name, structural_check):
    stats = create_backtest_stats(results)
    transactions = results[-1]
    output_file = os.path.join(os.getcwd(), "test-results", f"report-{indicator_name}-backtest.csv")

    mode = 'a' if os.path.exists(output_file) else 'w'
    
    with open(output_file, mode) as f: #
        if mode == 'w':
            print(
            "Ticker,Structural Check,Initial Capital,Final Capital,Total Return (%),Max Drawdown (%),Volatility (per step),Sharpe Ratio,Number of Steps,Peak Equity,Final Drawdown,Long Trades,Short Trades,Winner Longs,Winner Shorts,Loser Longs,Loser Shorts,", 
            file=f)    
        print(f"{stats['Ticker']},{structural_check},{stats['Initial Capital']:.2f},{stats['Final Capital']:.2f},{stats['Total Return (%)']:.2f},{stats['Max Drawdown (%)']:.2f},{stats['Volatility (per step)']:.4f},{stats['Sharpe Ratio']:.4f},{stats['Number of Steps']},{stats['Peak Equity']:.2f},{stats['Final Drawdown (%)']:.2f},{stats['Long Trades']},{stats['Short Trades']},{stats['Winner Longs']},{stats['Winner Shorts']},{stats['Loser Longs']},{stats['Loser Shorts']}", file=f)

        transactions_file = os.path.join(os.getcwd(), "test-results", f"report-{indicator_name}-{quote_name}-backtest-details.csv")
        os.remove(transactions_file) if os.path.exists(transactions_file) else None
        with open(transactions_file, 'w') as t: #
            print(
                "Ticker,Entry Index,Exit Index,Duration,Side,Entry Price,Exit Price,PL,Exit Reason,Friction,Entry Force, Θl↑,Θh↓,ER,ATR %,φ1,φ2,W,d", 
                file=t)
            for transaction in transactions:
                print(f"{transaction.ticker},{transaction.entry_index},{transaction.exit_index},{transaction.duration},{transaction.side},{transaction.entry_price:.2f},{transaction.exit_price:.2f},{transaction.pl:.2f},{transaction.exit_reason},{transaction.friction:.5f},{transaction.entry_force:.5f},{transaction.t_l_up:.5f},{transaction.t_h_dn:.5f},{transaction.efficiency_ratio:.5f},{transaction.current_atr_pct:.5f},{transaction.phi_1:.5f},{transaction.phi_2:.5f},{transaction.W:.5f},{transaction.d:.5f}", file=t)

def check_if_tradable(quote_stats):
    try:
        return quote_stats["Edge"] > 6
    except:
        return False
    
def get_stats(model_stats, quote_name):
    try:
        model_stats = model_stats.loc[quote_name]
        return model_stats
    except:
        return None

def already_traded(indicator_name, quote_name):
    transactions_file = os.path.join(os.getcwd(), "test-results", f"report-{indicator_name}-{quote_name}-backtest-details.csv")
    return os.path.exists(transactions_file)

def predict(quote_name, X_test):
    checkpoint_filepath = os.path.join(os.getcwd(), 'models', f'{quote_name}-{indicator_name}.keras')
    model = tf.keras.models.load_model(checkpoint_filepath, custom_objects={'directional_mse': directional_mse})    
    X_test = X_test.to_numpy().reshape((X_test.shape[0], X_test.shape[1], 1))    
    predictions = model.predict(X_test)
    return predictions

indicators = {
    "pricetime": wavelets.pricetime,
    "volumetime": wavelets.volumetime,
    "volumeprice": volumeprice,
    "ensemble": ensemble,
    "bardirection": bardirection
}


if __name__ == "__main__":
    quotes_file = sys.argv[1]
    indicator_name = sys.argv[2]    
    scale_multiplier = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    indicator = indicators[indicator_name]
    lookback_periods = 14
    _, sqlalchemy_url = db_config()
    quotes = read_quote_names(quotes_file)
    model_stats_file = os.path.join(os.getcwd(), "test-results", f"report-{indicator_name}.csv")
    model_stats = pd.read_csv(model_stats_file)
    model_stats.set_index('Ticker', inplace=True)

    for quote_name in quotes:                
        _, _, X_test, _, _, _, test_data = create_trade_datasets(indicator(sqlalchemy_url, quote_name, lookback_periods))
        quote_stats = get_stats(model_stats,quote_name)
        tradable = quote_stats is not None and check_if_tradable(quote_stats)
        
        try:
            if check_if_tradable(quote_stats) and not already_traded(indicator_name, quote_name):
                print(f"{quote_name} is tradable with {indicator_name} indicator.")
                predictions = predict(quote_name, X_test)
                edge = quote_stats["Edge"]     
                contrarian = np.sign(quote_stats["Match %"]-quote_stats["Diff %"]) < 0
                structural_check = False # True if indicator_name == "ensemble" else False
                scalping = False
                results = trade_quotes(sqlalchemy_url, quote_name, test_data, predictions, edge, contrarian, structural_check, scalping)
                generate_reports(results, indicator_name, quote_name, structural_check)
                print(f"Traded {len(test_data)} quotes from {quote_name} with {indicator_name} indicator and  tructural check = {structural_check}.")
            else:
                if already_traded(indicator_name, quote_name):
                    print(f"{quote_name} is already traded with {indicator_name} indicator.")
                else:
                    print(f"{quote_name} is not tradable with {indicator_name} indicator because edge is {quote_stats['Edge']}.")

        except ValueError as value_error:
            print(f"💀: Uncaught exception while trading {quote_name}")
            print(traceback.format_exc())

