from qf.indicators import price_time_wavelet_direction, price_time_wavelet_force, volume_time_wavelet_direction
import sys
import os
from qf.dbsync import read_quote_names, db_config
from qf.nn import directional_mse
from qf.nn import create_trade_datasets
import tensorflow as tf
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import traceback
from qf.trade import Position, Transaction
OMEGA_MAX = 0.20
FLOOR =0.13
CEILING = 0.18

# Import local classes (assuming they are in the same directory or python path)

def get_quotes(connection, quote_name, index):
    engine = create_engine(sqlalchemy_url)
    # Fetch OHLC + Volume to match X_test timestamps
    sql_template = f"""
        with quote_data as (
            SELECT "TIMESTAMP", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", ("CLOSE" - LAG("CLOSE", 1) over (ORDER BY "TIMESTAMP")) / ("CLOSE" + LAG("CLOSE", 1) over (ORDER BY "TIMESTAMP")) "ΔP"
            FROM QUOTE WHERE "TICKER" = '{quote_name}'
        ) select "TIMESTAMP", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "CLOSE", COALESCE(STDDEV( "ΔP") OVER(ORDER BY "TIMESTAMP" ROWS BETWEEN 30 PRECEDING AND CURRENT ROW),0.0001) "δP" 
        from quote_data
        order by "TIMESTAMP"
    """
    
    df = pd.read_sql(sql_template, connection)        
    df.set_index('TIMESTAMP', inplace=True)
    df = df.loc[df.index.intersection(index)] 
    df = df.sort_index()
    
    return df

def check_structure(direction, row):
    """
    Validates the Price-Time Geometry (Angles).
    Long: Higher Low (Θl↑) > Lower High (Θh↓)
    Short: Lower High (Θh↓) > Higher Low (Θl↑)
    """
    theta_l_up = row.get("Θl↑", 0)
    theta_h_dn = row.get("Θh↓", 0)
    
    # Structure Check
    if direction > 0: # Long
        return theta_l_up > theta_h_dn
    elif direction < 0: # Short
        return theta_h_dn > theta_l_up
    return False


def get_returns(equity_array):
    if len(equity_array) > 1:
        return np.diff(equity_array) / equity_array[:-1]
    else:
        return np.array([0])
    
def create_backtest_stats(quote_name, equity_curve, long_trades, short_trades, winner_longs, winner_shorts, loser_longs, loser_shorts, transactions):
    equity_array = np.ravel(equity_curve)
    prev_equity = equity_array[:-1]    

    # Initialize returns with 0
    returns = np.zeros_like(prev_equity)
    
    # Only calculate for indices where capital was still above zero
    valid_mask = prev_equity > 0
    returns[valid_mask] = np.diff(equity_array)[valid_mask] / prev_equity[valid_mask]
    
    volatility = np.std(returns) if len(returns) > 0 else 0
    # If final_capital was an array due to the previous bug, take the first value
    f_cap = equity_array[-1]
    
    initial_capital = equity_array[0]
    total_return_pct = (f_cap - initial_capital) / initial_capital
    
    # Now np.diff will produce a simple 1D vector
    returns = get_returns(equity_array)
    
    # Standard deviation of returns (volatility)
    volatility = np.std(returns) if len(returns) > 0 else 0
    
    # Sharpe Ratio (Assuming 0 risk-free rate for simplicity)
    # Annualization factor depends on the timeframe of your data (e.g., np.sqrt(252))
    sharpe_ratio = (np.mean(returns) / volatility) if volatility != 0 else 0

    # 3. Drawdown Analysis
    # Peak equity reached up to each point in time
    running_max = np.maximum.accumulate(equity_array)
    drawdowns = (equity_array - running_max) / running_max
    max_drawdown = np.min(drawdowns)

    # 4. Summary Statistics
    stats = {
        "Ticker": quote_name,
        "Initial Capital": initial_capital,
        "Final Capital": f_cap,
        "Total Return (%)": total_return_pct * 100,
        "Max Drawdown (%)": max_drawdown * 100,
        "Volatility (per step)": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Number of Steps": len(equity_array),
        "Peak Equity": np.max(equity_array),
        "Final Drawdown (%)": drawdowns[-1] * 100,
        "Long Trades": long_trades,
        "Short Trades": short_trades,
        "Winner Longs": winner_longs,
        "Winner Shorts": winner_shorts,
        "Loser Longs": loser_longs,
        "Loser Shorts": loser_shorts
    }
    return stats, transactions

def get_stats_params(quote_name, force_stats):
    """
    Retrieves MAE and Edge from the report file for the given ticker.
    """
    filename = 'report-price-time-wavelet-direction.csv'
    default_mae = 0.02
    default_edge = 10.0
    
    if not os.path.exists(filename):
        return default_mae, default_edge
    
    try:
        df = force_stats
        row = df[df['Ticker'] == quote_name]
        if not row.empty:
            mae = row.iloc[0]['MAE']
            edge = row.iloc[0]['Edge']
            return mae, edge
    except Exception as e:
        print(f"Warning: Could not read stats for {quote_name}: {e}")
        pass
    return default_mae, default_edge

def calculate_levels(current_price, signal, δP):
    """
    Calculates dynamic SL and TP based on real-time volatility (delta_p).
    """
    # Enforce scalars
    cp = float(current_price)
    dp = float(δP)
    
    sl_multiplier = 2
    tp_multiplier = 3
    
    sl_distance = cp * (dp * sl_multiplier)
    tp_distance = cp * (dp * tp_multiplier)

    if signal > 0:  # Long
        stop_loss = cp - sl_distance
        take_profit = cp + tp_distance
    else:  # Short
        stop_loss = cp + sl_distance
        take_profit = cp - tp_distance

    return float(stop_loss), float(take_profit)

def update_hybrid_exit(position, row, delta_p, v_dir):
    """
    Updates SL using trailing logic and checks intra-bar extremes (HIGH/LOW).
    """
    # Force everything to standard floats to avoid Series comparison errors
    high = float(row['HIGH'])
    low = float(row['LOW'])
    # Handle possible Series/numpy input for CLOSE
    close_val = row['CLOSE']
    close = float(close_val.to_numpy()[0]) #  if hasattr(close_val, 'to_numpy') else close_val
    dp = float(delta_p)
    
    trail_dist = close * (dp * 2.0)

    if position.side == 1:  # LONG
        # 1. Check for Exit via intra-bar extremes
        if low <= position.stop_loss:
            return None, float(position.stop_loss)
        elif high >= position.take_profit:
            return None, float(position.take_profit)
        elif v_dir != position.side and close > position.entry_price:
            return None, float(close)
        
        # 2. Update trailing SL
        new_sl_calc = close - trail_dist
        updated_sl = max(float(position.stop_loss), new_sl_calc)
        return position._replace(stop_loss=updated_sl), None
            
    else:  # SHORT
        # 1. Check for Exit
        if high >= position.stop_loss:
            return None, float(position.stop_loss)
        elif low <= position.take_profit:
            return None, float(position.take_profit)
        elif v_dir != position.side and position.entry_price > close:
            return None, float(close)
                   
        # 2. Update trailing SL
        new_sl_calc = close + trail_dist
        updated_sl = min(float(position.stop_loss), new_sl_calc)
        return position._replace(stop_loss=updated_sl), None

def trade_quotes(quote_name, df, price_time_predictions, volume_time_predictions, force_predictions):
    initial_capital = 10000.0
    current_capital = float(initial_capital)
    active_position = None
    transactions = []
    
    long_trades, short_trades = 0, 0
    winner_longs, winner_shorts = 0, 0
    loser_longs, loser_shorts = 0, 0
    equity_curve = []

    for i in range(len(df)):
        row = df.iloc[i]
        curr_dp = float(row['δP'])
        p_dir = int(np.sign(price_time_predictions[i][0]))
        v_dir = int(np.sign(volume_time_predictions[i][0]))
        f_val = force_predictions[i][0]
        close_series = row['CLOSE']
        curr_close = float(close_series.to_numpy()[0] if hasattr(close_series, 'to_numpy') else close_series)

        # --- 1. EXIT LOGIC (Trailing Stop Update) ---
        if active_position is not None:
            updated_pos, exit_price = update_hybrid_exit(active_position, row, curr_dp, v_dir)
            
            if exit_price is not None:
                pl_total = float((exit_price - active_position.entry_price) * active_position.side * active_position.quantity)
                current_capital += pl_total
                
                if active_position.side == 1:
                    long_trades += 1
                    if pl_total > 0: winner_longs += 1
                    else: loser_longs += 1
                else:
                    short_trades += 1
                    if pl_total > 0: winner_shorts += 1
                    else: loser_shorts += 1
                
                transactions.append(Transaction.from_position(active_position, pl_total, i, exit_price))
                active_position = None
            else:
                active_position = updated_pos

        # --- 2. ENTRY LOGIC (Force Scaling) ---
        if active_position is None:
            confidence = (f_val - FLOOR) / (CEILING - FLOOR)

            if v_dir != p_dir: # and check_structure(p_dir, row):
                dynamic_qty = max(10, int(100 * confidence))                
                sl, tp_base = calculate_levels(curr_close, p_dir, curr_dp)                                    
                tp_dist = abs(tp_base - curr_close) * (1.0 + (0.5 * confidence))
                final_tp = curr_close + (tp_dist * p_dir)

                active_position = Position(
                    ticker=quote_name,
                    entry_index=i,
                    entry_price=curr_close,
                    entry_force=f_val,
                    side= v_dir,
                    quantity=dynamic_qty,
                    take_profit=float(final_tp),
                    stop_loss=float(sl)
                )

        # --- 3. EQUITY TRACKING ---
        unrealized = 0.0
        if active_position:
            unrealized = float((curr_close - active_position.entry_price) * active_position.side * active_position.quantity)
        
        equity_curve.append(float(current_capital + unrealized))

    return create_backtest_stats(
        quote_name, equity_curve, long_trades, short_trades, 
        winner_longs, winner_shorts, loser_longs, loser_shorts, transactions
    )

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
    if not os.path.exists(checkpoint_filepath):
        print(f"Warning: Model {checkpoint_filepath} not found.")
        return np.zeros((len(X_test), 1))
        
    model = tf.keras.models.load_model(checkpoint_filepath, custom_objects={'directional_mse': directional_mse})    
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

    # Load Reports
    price_direction_stats = get_model_stats(os.getcwd(), "report-price-time-wavelet-direction.csv")
    volume_direction_stats = get_model_stats(os.getcwd(), "report-volume-time-wavelet-direction.csv")
    force_stats = get_model_stats(os.getcwd(), "report-price-time-wavelet-force.csv")
    lookback_periods = 14
    engine = create_engine(sqlalchemy_url)

    with engine.connect() as connection:
        for quote_name in quotes:
            price_direction = get_stats(price_direction_stats, quote_name)
            volume_direction = get_stats(volume_direction_stats, quote_name)
            force_stats = get_stats(force_stats, quote_name)        
            tradable = check_if_tradable(price_direction) and check_if_tradable(volume_direction)
            
            if tradable:
                # Create datasets
                _, _, X_price_time_test, _, _, _, _ = create_trade_datasets(price_time_wavelet_direction(connection, quote_name, lookback_periods))
                _, _, X_volume_time_test, _, _, _, _ = create_trade_datasets(volume_time_wavelet_direction(connection, quote_name, lookback_periods))
                _, _, X_force_test, _, _, _, _ = create_trade_datasets(price_time_wavelet_force(connection, quote_name, lookback_periods))
                
                if not (len(X_volume_time_test) == len(X_price_time_test) and len(X_volume_time_test) == len(X_force_test)):
                    print(f"Data length mismatch for {quote_name}: Price-Time={len(X_price_time_test)}, Volume-Time={len(X_volume_time_test)}, Force={len(X_force_test)}. Skipping.")
                    continue
                else:
                    print(f"Data length match for {quote_name}: {len(X_price_time_test)} samples. Proceeding with backtest.")
                
                # Get Predictions
                price_time_predictions = predict(quote_name, "price-time-wavelet-direction", X_price_time_test)
                volume_time_predictions = predict(quote_name, "volume-time-wavelet-direction", X_volume_time_test)
                force_predictions = predict(quote_name, "price-time-wavelet-force", X_force_test)

                # Get Quotes
                quotes = get_quotes(connection, quote_name, X_price_time_test.index)
                assert len(quotes) == len(X_price_time_test)

                stats, transactions = trade_quotes(
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
                        f"{stats['Ticker']}, {stats['Initial Capital']}, {stats['Final Capital']}, {stats['Total Return (%)']:.2f}, {stats['Max Drawdown (%)']:.2f}, {stats['Volatility (per step)']:.2f}, {stats['Sharpe Ratio']:.2f}, {stats['Number of Steps']}, {stats['Peak Equity']:.2f}, {stats['Final Drawdown (%)']:.2f}, {stats['Long Trades']}, {stats['Short Trades']}, {stats['Winner Longs']}, {stats['Winner Shorts']}, {stats['Loser Longs']}, {stats['Loser Shorts']}",
                        file=f
                    )     
          

                output_file = os.path.join(os.getcwd(), "test-results", f"report-{quote_name}-transactions.csv")
                with open(output_file, mode) as f:
                    print(
                        "Entry Index, Entry Price, Entry Force, Side, Quantity, Take Profit, Stop Loss, PL, Exit Index, Exit Price",
                        file=f
                    )    

                    for transaction in transactions:
                        print(
                            f"{transaction.entry_index}, {transaction.entry_price}, {transaction.entry_force}, {transaction.side}, {transaction.quantity}, {transaction.take_profit}, {transaction.stop_loss}, {transaction.pl}, {transaction.exit_index}, {transaction.exit_price}",
                            file=f
                        )             

            else:
                print(f"Skipping {quote_name} (Not Tradable or No Data)")
        connection.close()
    engine.dispose()