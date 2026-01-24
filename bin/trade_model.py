from bin.train_model import directional_mse
from qf.indicators import volumeprice
from qf.indicators import ensemble
from qf.indicators import wavelets
from qf.trade.stats import create_backtest_stats
import sys
import os
from qf.dbsync import read_quote_names, db_config
from qf.indicators import scale_with_multiplier
from qf.nn.splitter import create_trade_datasets
from qf.trade import Position, Transaction
from qf.trade.stats import get_returns
import tensorflow as tf
import pandas as pd
import qf.trade as trade
from sqlalchemy import create_engine
import numpy as np
import traceback

indicator = {
    "pricetime": wavelets.pricetime,
    "volumetime": wavelets.volumetime,
    "volumeprice": volumeprice,
    "ensemble": ensemble
}

def get_quotes(sqlalchemy_url, quote_name, X_test):
    engine = create_engine(sqlalchemy_url)
    sql_template = f"SELECT \"TIMESTAMP\", \"OPEN\", \"HIGH\", \"LOW\", \"CLOSE\" FROM QUOTE WHERE  \"TICKER\" = '{quote_name}'"
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

def trade_quotes(sqlalchemy_url, quote_name, context, predictions, edge, initial_cap = 10000):
    transaction_log = []    
    cash = [initial_cap]
    longs, shorts = 0, 0
    winner_longs, winner_shorts = 0, 0
    loser_longs, loser_shorts = 0, 0
    timestamp = context.index
    position = None
    n = len(context)
    context = get_quotes(sqlalchemy_url, quote_name, context)
    
    assert  len(context) == len(predictions)

    for i in range(n-1):
        t = timestamp[i]
        prediction = predictions[i]
        open_price, close_price, high_price, low_price = context.loc[t, "OPEN"], context.loc[t, "CLOSE"], context.loc[t, "HIGH"], context.loc[t, "LOW"]
        transaction = Transaction.from_position(position, i, n, open_price, low_price, high_price, close_price)

        if not transaction is None:            
            if transaction.exit_reason == 1:
                winner_longs += 1 if transaction.side == 1 else 0
                winner_shorts += 1 if transaction.side == -1 else 0
            elif transaction.exit_reason == -1:
                loser_longs += 1 if transaction.side == 1 else 0
                loser_shorts += 1 if transaction.side == -1 else 0
            transaction_log.append(transaction)
            cash.append(cash[-1] + transaction.pl)
            position = None
        else:
            cash.append(cash[-1])

        if position is None:
            entry_index = i + 1
            t_1 = timestamp[entry_index]            
            entry_price = context.loc[t_1, "OPEN"]                       
            position = Position.create(quote_name, context, entry_index, prediction, edge, entry_price)
            if position.side == 1:
                longs += 1
            else:
                shorts += 1
            
        
    print(f"Traded {len(transaction_log)} transactions for {quote_name}")
    return (quote_name, cash, longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts, transaction_log)

def predict(quote_name, X_test):
    checkpoint_filepath = os.path.join(os.getcwd(), 'models', f'{quote_name}-{indicator_name}.keras')
    model = tf.keras.models.load_model(checkpoint_filepath, custom_objects={'directional_mse': directional_mse})
    X_test = X_test.to_numpy().reshape((X_test.shape[0], X_test.shape[1], 1))    
    predictions = model.predict(X_test)
    return predictions


def generate_reports(results, indicator_name, quote_name):
    stats = create_backtest_stats(results)
    transactions = results[-1]
    output_file = os.path.join(os.getcwd(), "test-results", f"report-{indicator_name}-backtest.csv")

    mode = 'a' if os.path.exists(output_file) else 'w'
    
    with open(output_file, mode) as f: #
        if mode == 'w':
            print(
            "Ticker,Initial Capital,Final Capital,Total Return (%),Max Drawdown (%),Volatility (per step),Sharpe Ratio,Number of Steps,Peak Equity,Final Drawdown,Long Trades,Short Trades,Winner Longs,Winner Shorts,Loser Longs,Loser Shorts,", 
            file=f)    
        print(f"{stats['Ticker']},{stats['Initial Capital']:.2f},{stats['Final Capital']:.2f},{stats['Total Return (%)']:.2f},{stats['Max Drawdown (%)']:.2f},{stats['Volatility (per step)']:.4f},{stats['Sharpe Ratio']:.4f},{stats['Number of Steps']},{stats['Peak Equity']:.2f},{stats['Final Drawdown (%)']:.2f},{stats['Long Trades']},{stats['Short Trades']},{stats['Winner Longs']},{stats['Winner Shorts']},{stats['Loser Longs']},{stats['Loser Shorts']}", file=f)

        transactions_file = os.path.join(os.getcwd(), "test-results", f"report-{indicator_name}-{quote_name}-backtest-details.csv")
        os.remove(transactions_file) if os.path.exists(transactions_file) else None
        with open(transactions_file, 'w') as t: #
            print(
                "Ticker,Entry Index,Exit Index,Duration,Side,Entry Price,Exit Price,PL,TP Price,SL Price,Exit Reason", 
                file=t)
            for transaction in transactions:
                print(f"{transaction.ticker},{transaction.entry_index},{transaction.exit_index},{transaction.duration},{transaction.side},{transaction.entry_price:.2f},{transaction.exit_price:.2f},{transaction.pl:.2f},{transaction.take_profit:.2f},{transaction.stop_loss:.2f},{transaction.exit_reason}", file=t)

def check_if_tradable(quote_stats):
    try:
        return quote_stats["tradable"]
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

if __name__ == "__main__":
    quotes_file = sys.argv[1]
    indicator_name = sys.argv[2]    
    scale_multiplier = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    indicator = indicator[indicator_name]
    lookback_periods = 14
    _, sqlalchemy_url = db_config()
    quotes = read_quote_names(quotes_file)
    model_stats_file = os.path.join(os.getcwd(), "test-results", f"report-{indicator_name}.csv")
    model_stats = pd.read_csv(model_stats_file)
    model_stats.set_index('Ticker', inplace=True)

    for quote_name in quotes:                
        _, _, X_test, _, _, _ = create_trade_datasets(scale_with_multiplier(indicator(sqlalchemy_url, quote_name, lookback_periods), scale_multiplier))
       
        quote_stats = get_stats(model_stats,quote_name)
        tradable = quote_stats is not None and check_if_tradable(quote_stats)
        
        try:
            if check_if_tradable(quote_stats) and not already_traded(indicator_name, quote_name):
                print(f"{quote_name} is tradable with {indicator_name} indicator.")
                predictions = predict(quote_name, X_test)
                edge = quote_stats["Edge"]            
                results = trade_quotes(sqlalchemy_url, quote_name, X_test, predictions, edge)
                generate_reports(results, indicator_name, quote_name)
                print(f"Traded {len(X_test)} quotes from {quote_name} with {indicator_name} indicator.")
            else:
                if already_traded(indicator_name, quote_name):
                    print(f"{quote_name} is already traded with {indicator_name} indicator.")
                else:
                    print(f"{quote_name} is not tradable with {indicator_name} indicator.")

        except ValueError as value_error:
            print(f"💀: Uncaught exception while trading {quote_name}")
            print(traceback.format_exc())

