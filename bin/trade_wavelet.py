import json
from qf.indicators import price_time_wavelet_direction, price_time_wavelet_force, volume_time_wavelet_direction
import sys
import os
from qf.dbsync import read_quote_names, db_config
from qf.nn import directional_mse
from qf.nn import create_trade_datasets
from qf.trade import trade_forex, trade_stocks
import tensorflow as tf
import pandas as pd
from sqlalchemy import create_engine
from qf.trade import trade_forex
from qf.trade import trade_stocks

def get_quotes(connection, quote_name, index):
    # Fetch OHLC + Volume to match X_test timestamps
    sql_template = f"""
        with quote_data as (
            SELECT "TICKER", "TIMESTAMP", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", 
            ("CLOSE" - LAG("CLOSE", 1) over (ORDER BY "TIMESTAMP")) / ("CLOSE" + LAG("CLOSE", 1) over (ORDER BY "TIMESTAMP")) "ΔP"  
            FROM "QUOTE" WHERE "TICKER" = '{quote_name}'
        ) select quote_data."TICKER", quote_data."TIMESTAMP", "OPEN", "HIGH", "LOW", "CLOSE", 
            COALESCE(STDDEV(quote_data."ΔP") OVER(ORDER BY quote_data."TIMESTAMP" ROWS BETWEEN 30 PRECEDING AND CURRENT ROW),0.0001) "δP" ,
            coalesce(POWER((AVG("VOLUME") OVER(ORDER BY quote_data."TIMESTAMP" ROWS BETWEEN 30 PRECEDING AND CURRENT ROW) - "VOLUME") / (AVG("VOLUME") OVER(ORDER BY quote_data."TIMESTAMP" ROWS BETWEEN 30 PRECEDING AND CURRENT ROW) + "VOLUME" + 0.000009),2),0) "V",
            AVG("VOLUME") OVER(ORDER BY quote_data."TIMESTAMP" ROWS BETWEEN 30 PRECEDING AND CURRENT ROW) "AV",
            indicators."H"
        from quote_data inner join ANGULAR_INDICATORS('{quote_name}') indicators on quote_data."TIMESTAMP" = indicators."TIMESTAMP"
        order by "TIMESTAMP" 
    """
    
    df = pd.read_sql(sql_template, connection)        
    df.set_index('TIMESTAMP', inplace=True)
    df = df.loc[df.index.intersection(index)] 
    df = df.sort_index()
    
    return df


def check_if_tradable(quote_stats):
    try:
        return quote_stats["Edge"] > 10 and quote_stats["tradable"]
    except:
        return False
    
def get_stats(model_stats, quote_name):
    try:
        return model_stats.loc[quote_name]
    except:
        return None

def get_model_stats(current_dir, filename):
    model_stats_file = os.path.join(current_dir, "test-results", filename)
    if not os.path.exists(model_stats_file):
        print(f"Error: Report file {filename} not found.")
        return None
        
    model_stats = pd.read_csv(model_stats_file)
    # Clean whitespace from column names if any
    model_stats.columns = model_stats.columns.str.strip()
    model_stats.set_index('Ticker', inplace=True)
    return model_stats

def predict(quote_name, model_name, X_test):
    checkpoint_filepath = os.path.join(os.getcwd(), 'models', f'{quote_name}-{model_name}.keras')    
    try:
        if not os.path.exists(checkpoint_filepath):
            print(f"Warning: Model {checkpoint_filepath} not found.")
            return None

        model = tf.keras.models.load_model(checkpoint_filepath, custom_objects={'directional_mse': directional_mse})    
    except Exception as e:        
        print(f"Error loading model: {e}")
        return None
    
    # Ensure input shape is (N, Timesteps, 1) or (N, Features) depending on model
    # The CNN model in train_model.py expects (batch, lags, 1)
    X_input = X_test.to_numpy().reshape((X_test.shape[0], X_test.shape[1], 1))    
    predictions = model.predict(X_input, verbose=0)
    return predictions

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python trade_angular.py <quotes_file>")
        sys.exit(1)
        
    quotes_file = sys.argv[1]
    _, sqlalchemy_url = db_config()
    
    try:
        quotes = read_quote_names(quotes_file)
    except Exception as e:
        print(f"Error reading quotes: {e}")
        quotes = []

    # Load Reports.
    price_direction_stats = get_model_stats(os.getcwd(), "report-price-time-wavelet-direction.csv")
    volume_direction_stats = get_model_stats(os.getcwd(), "report-volume-time-wavelet-direction.csv")
    force_stats = get_model_stats(os.getcwd(), "report-price-time-wavelet-force.csv")
    lookback_periods = 14
    engine = create_engine(sqlalchemy_url)
    exit_reasons = { '-1': 'Stop Loss', '0': 'Early Stop', '1': 'Take Profit'}

    with engine.connect() as connection:
        for quote_name in quotes:
            price_direction = get_stats(price_direction_stats, quote_name)
            volume_direction = get_stats(volume_direction_stats, quote_name)
            force_stats = get_stats(force_stats, quote_name)        
            tradable = check_if_tradable(price_direction) and check_if_tradable(volume_direction)
            
            if tradable:
                details_file = os.path.join(os.getcwd(), "test-results", f"report-{quote_name}-transactions.json")
                if os.path.exists(details_file):
                    continue

                # Create datasets
                _, _, X_price_time_test, _, _, _, _ = create_trade_datasets(price_time_wavelet_direction(connection, quote_name, lookback_periods))
                _, _, X_volume_time_test, _, _, _, _ = create_trade_datasets(volume_time_wavelet_direction(connection, quote_name, lookback_periods))
                _, _, X_force_test, _, _, _, _ = create_trade_datasets(price_time_wavelet_force(connection, quote_name, lookback_periods))
                
                if len(X_force_test) < 60:
                    print(f"Insufficient data for trading with {quote_name}. Skipping.")
                    continue
                elif not (len (X_volume_time_test) == len(X_price_time_test) and len(X_volume_time_test) == len(X_force_test)):
                    X_force_test = X_force_test.reindex(X_volume_time_test.index)
                    print(f"Data length mismatch for {quote_name}: Price-Time={len(X_price_time_test)}, Volume-Time={len(X_volume_time_test)}, Force={len(X_force_test)}. Adju.")
                else:
                    print(f"Data length match for {quote_name}: {len(X_price_time_test)} samples. Proceeding with backtest.")
                
                # Get Predictions
                if len(X_price_time_test) == 0 or len(X_volume_time_test) == 0 or len(X_force_test) == 0:
                    print(f"No data for {quote_name}. Skipping.")
                    continue

                price_time_predictions = predict(quote_name, "price-time-wavelet-direction", X_price_time_test)
                volume_time_predictions = predict(quote_name, "volume-time-wavelet-direction", X_volume_time_test)
                force_predictions = predict(quote_name, "price-time-wavelet-force", X_force_test)

                if price_time_predictions is None or volume_time_predictions is None or force_predictions is None:
                    print(f"No predictions for {quote_name}. Skipping.")
                    continue

                # Get Quotes
                quotes = get_quotes(connection, quote_name, X_price_time_test.index)
                assert len(quotes) == len(X_price_time_test)

                if quote_name.endswith("=X") and "JPY" not in quote_name:
                    # Use the specialized Forex function
                    stats, transactions = trade_forex(
                        quote_name, 
                        quotes,                                             
                        price_time_predictions, 
                        volume_time_predictions, 
                        force_predictions
                    )
                else:
                    # Use the existing standard/stock function
                    stats, transactions = trade_stocks(
                        quote_name, 
                        quotes,                                             
                        price_time_predictions, 
                        volume_time_predictions, 
                        force_predictions
                    )
                    
                output_file = os.path.join(os.getcwd(), "test-results", f"report-backtest.csv")
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
                        exit_reason = exit_reasons.get(str(transaction.exit_reason), 'Unknown')
                        transaction = {
                            "Entry Index": transaction.entry_index,
                            "Entry Price": float(transaction.entry_price),
                            "Entry Force": float(transaction.entry_force),
                            "Side": int(transaction.side),
                            "Quantity": int(transaction.quantity),
                            "Take Profit": float(transaction.take_profit),
                            "Stop Loss": float(transaction.stop_loss),
                            "PL": float(transaction.pl),
                            "Exit Index": int(transaction.exit_index),
                            "Exit Price": float(transaction.exit_price),
                            "Exit Reason": exit_reason,
                            "position_history": [{"index": s.index, "open_price": float(s.open_price), "high_price": float(s.high_price), "low_price": float(s.low_price), "close_price": float(s.close_price), "dP": float(s.δP), "V": float(s.V), "H": float(s.H), "previous_force": float(s.previous_force), "current_force": float(s.current_force)} for s in transaction.state]
                        }
                        transaction_list.append(transaction)
                    print(json.dumps(transaction_list), file=f)
            else:
                print(f"Skipping {quote_name} (Not Tradable or No Data)")
        connection.close()
    engine.dispose()