import json
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
from qf.trade import Position, Transaction
import math
from qf.trade.State import State

OMEGA_MAX = 0.20
FLOOR =0.13
CEILING = 0.18

# Import local classes (assuming they are in the same directory or python path)

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

def calculate_levels(current_price, signal, delta_p, H, V, is_forex, is_jpy):
    cp = float(current_price)
    dp = float(delta_p)
    
    # Preserve Volume scaling logic
    v_factor = math.sqrt(max(0, float(V)))
    v_multiplier = 1.0 + min(v_factor, 2.0) 

    # Preserve experimental parameters
    SL_TINY_BASE = 0.40 
    TP_BASE = 1.0
    H_SCALAR = 7.5
    
    # Calculate Volatility-based distance
    vol_sl_distance = cp * dp * SL_TINY_BASE * v_multiplier
    
    # LOOPHOLE FIX: Structural Floor
    # Prevents sl_distance from becoming too small, which caps max leverage
    sl_floor_pct = 0.001 if is_forex else 0.005
    sl_floor_distance = cp * sl_floor_pct
    
    # Take the larger of the two to ensure a safe risk-denominator
    sl_distance = max(vol_sl_distance, sl_floor_distance)
    
    # Preserve Take Profit logic with Force (H) scaling
    tp_multiplier = (TP_BASE + (H_SCALAR * float(H))) * v_multiplier
    tp_distance = cp * dp * tp_multiplier

    tick_size = 0.01 if is_jpy else (0.0001 if is_forex else 0.01)

    if signal == 1:  # Long
        tp = current_price + tp_distance
        sl = current_price - sl_distance
        final_tp = math.floor(tp / tick_size) * tick_size
        final_sl = math.floor(sl / tick_size) * tick_size
    else:            # Short
        tp = current_price - tp_distance
        sl = current_price + sl_distance
        final_tp = math.ceil(tp / tick_size) * tick_size
        final_sl = math.ceil(sl / tick_size) * tick_size
        
    return float(final_tp), float(final_sl)

        
import math

def update_hybrid_exit(position, row, delta_p, current_force, is_forex):
    """
    Consolidated Exit Logic:
    1. Removes quantity loophole via Structural Risk Floor.
    2. Syncs multipliers with calculate_levels (SL_TINY_BASE = 0.40).
    3. Implements Dynamic Slippage based on volatility (dp).
    4. Preserves Early Stops (Momentum Decay & Break-even).
    """
    high = float(row['HIGH'])
    low = float(row['LOW'])
    close = float(row['CLOSE'])
    ticker = str(row['TICKER'])
    dp = float(delta_p)
    v_val = float(row['V'])
    
    is_jpy = "JPY" in ticker    
    tick_size = 0.01 if is_jpy else (0.0001 if is_forex else 0.01)

    # REFINEMENT: Dynamic Slippage & Fees
    # Scales slippage based on 10% of the current standard deviation (dp)
    slippage_cost = close * dp * 0.1 
    exit_fee_per_share = 0.005 if not is_forex else 0.0 
    profit_dist = (close - position.entry_price) * position.side

    def snap_to_tick(price, side):
        if side == 1: return math.floor(price / tick_size) * tick_size
        else: return math.ceil(price / tick_size) * tick_size

    # BENEFIT 1: Momentum Decay Exit (Early Stop)
    # If in profit and Force drops 30% from entry, exit immediately.
    momentum_trigger = (10 * tick_size) if is_forex else (position.entry_price * 0.001)
    if profit_dist > momentum_trigger and current_force < (position.entry_force * 0.70):
        exit_price_raw = close - (slippage_cost * position.side) 
        return None, snap_to_tick(exit_price_raw, position.side) - exit_fee_per_share, 0

    # BENEFIT 2: Break-even Protection
    # Locks in minor profit if the price moves significantly in our favor
    be_trigger = (6 * tick_size) if is_forex else (position.entry_price * 0.0006)
    if profit_dist > be_trigger:
        exit_price_raw = close - (slippage_cost * position.side)
        # Note: '0' maps to 'Early Stop' in your report dictionary
        return None, snap_to_tick(exit_price_raw, position.side) - exit_fee_per_share, 0

    # LOOPHOLE FIX: Synchronize Trailing Logic with Entry Floor
    # Use the exact same multipliers as calculate_levels
    v_multiplier = 1.0 + min(math.sqrt(max(0, v_val)), 2.0)
    sl_floor_pct = 0.001 if is_forex else 0.005
    
    # Calculate the minimum safe trailing distance
    vol_stop_dist = close * dp * 0.40 * v_multiplier
    floor_stop_dist = close * sl_floor_pct
    safe_trailing_dist = max(vol_stop_dist, floor_stop_dist)

    if position.side == 1:  # LONG
        # Check Hard Stops First
        if low <= position.stop_loss:
            return None, float(position.stop_loss) - exit_fee_per_share, -1
        elif high >= position.take_profit:
            return None, float(position.take_profit) - exit_fee_per_share, 1
            
        # Update Trailing Stop
        new_sl_raw = close - safe_trailing_dist
        # Optional: Aggressive trail if deep in profit
        if profit_dist > (dp * position.entry_price * 0.75):
            new_sl_raw = max(new_sl_raw, position.entry_price + (tick_size * 2))
            
        updated_sl = max(float(position.stop_loss), math.floor(new_sl_raw / tick_size) * tick_size)
        return position._replace(stop_loss=updated_sl), None, 0
            
    else:  # SHORT
        # Check Hard Stops First
        if high >= position.stop_loss:
            return None, float(position.stop_loss) - exit_fee_per_share, -1
        elif low <= position.take_profit:
            return None, float(position.take_profit) - exit_fee_per_share, 1
            
        # Update Trailing Stop
        new_sl_raw = close + safe_trailing_dist
        if profit_dist > (dp * position.entry_price * 0.75):
            new_sl_raw = min(new_sl_raw, position.entry_price - (tick_size * 2))
            
        updated_sl = min(float(position.stop_loss), math.ceil(new_sl_raw / tick_size) * tick_size)
        return position._replace(stop_loss=updated_sl), None, 0

def calculate_liquidity_cap(H, V, minimum):
    """
    H: Volume Wavelet Force (Geometric concentration).
    V: Squared standardized volume (Instability/Noise).
    minimum: The base lot size (anchor).
    
    Returns a capacity that scales the 'minimum' by the structural quality.
    """
    # H_ref is the force threshold (0.13) where we take the base lot.
    H_ref = 0.13
    
    # Geometric quality: Higher H and Lower V increase capacity.
    # We use sqrt(V) to get back to the Z-score magnitude scale.
    quality = (H / H_ref) / (1 + np.sqrt(V))
    
    return int(minimum * quality)

def calculate_dynamic_qty(confidence, H, V, current_capital, entry_price, stop_loss, is_forex, min_qty=1, max_qty=1000, max_risk_pct=0.02, max_cap_pct=0.03):
    """
    Compounds capital using Volatility-Targeting.
    Constrained by:
    1. A hard 2% Risk-of-Capital Rule (Stop Loss based)
    2. A hard Capital Exposure Limit (Notional based)
    """
    # 1. Calculate Absolute Stop Loss Distance (Volatility Risk)
    sl_distance = abs(entry_price - stop_loss)
    if sl_distance <= 0:
        sl_distance = entry_price * 0.001
        
    # 2. Performance-Based Conviction (Unchanged compounding logic)
    conviction = (confidence * 0.6) + (H * 0.2) + (abs(V) * 0.2) if H > 0 and V > 0 else confidence
    conviction = max(0.1, min(1.0, conviction))
    
    # 3. The Risk Constraint (Dollar Risk Calculation)
    # This limits how much is LOST if the stop is hit.
    target_risk_pct = max_risk_pct * conviction
    dollar_risk = current_capital * target_risk_pct
    
    # 4. Volatility-Adjusted Quantity
    qty_by_risk = int(dollar_risk / sl_distance)
    
    # 5. NEW: Capital Exposure Constraint (The "2% Capital Limit")
    # This limits how much is SPENT to open the position.
    # For FOREX, we still allow leverage (e.g., max_leverage of 5.0)
    # For Stocks, we cap it at your max_cap_pct (e.g., 0.02 for 2%)
    if is_forex:
        max_leverage = 5.0 
        max_allowed_notional = current_capital * max_leverage
    else:
        # This enforces your "min 2% capital limit" correctly
        max_allowed_notional = current_capital * max_cap_pct

    qty_by_cap = int(max_allowed_notional / entry_price)
    
    # 6. Apply all constraints
    # Takes the smaller of risk-based size or capital-limit-based size
    final_qty = min(qty_by_risk, qty_by_cap)
    
    # 7. LIQUIDITY CONSTRAINTS (Unchanged)
    if final_qty < min_qty:
        return 0
        
    final_qty = min(final_qty, max_qty)
    
    return final_qty

def trade_quotes(quote_name, df, price_time_predictions, volume_time_predictions, force_predictions):    
    initial_capital = 10000.0
    current_capital = float(initial_capital)
    active_position = None
    transactions = []
    
    long_trades, short_trades = 0, 0
    winner_longs, winner_shorts = 0, 0
    loser_longs, loser_shorts = 0, 0
    equity_curve = []
    is_forex = quote_name.endswith("=X")
    is_jpy = "JPY" in quote_name
    current_floor = FLOOR * 0.9 if is_forex else FLOOR + 0.02
    current_ceiling = CEILING * 1.1 if is_forex else CEILING

    for i in range(len(df)):
        row = df.iloc[i]
        curr_dp = float(row['δP'])
        p_dir = int(np.sign(price_time_predictions[i][0]))
        v_dir = int(np.sign(volume_time_predictions[i][0]))
        f_val = force_predictions[i][0]
        close_series = row['CLOSE']
        H = row['H']
        V = row['V']        
        effective_dir =  p_dir if is_forex else (v_dir if v_dir != p_dir else 0)
        effective_V = V if not is_forex else 0        
        effective_H = H if not is_forex else 0        
        curr_close = float(close_series.to_numpy()[0] if hasattr(close_series, 'to_numpy') else close_series)

        # --- 1. EXIT LOGIC (Trailing Stop Update) ---
        if not (active_position is None):
            updated_pos, exit_price , exit_reason = update_hybrid_exit(
                active_position, 
                row, 
                curr_dp, 
                f_val,
                is_forex
            )
            
            if not (exit_price is None):
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

                transactions.append(Transaction.from_position(active_position, pl_total, i, exit_price, exit_reason))
                active_position = None
            else:
                active_position = updated_pos

        # --- 2. ENTRY LOGIC (Force Scaling) ---
        if (active_position is None) and ((not is_forex) or (is_forex and current_floor < f_val < current_ceiling)):
            confidence = (f_val - current_floor) / (current_ceiling - current_floor)

            if effective_dir != 0:
                # 1. Calculate Levels FIRST to get the Stop Loss (Volatility metric)
                tp_base, sl = calculate_levels(curr_close, effective_dir, curr_dp, effective_H, effective_V, is_forex, is_jpy)
                
                # 2. Calculate Quantity SECOND (Using the compounding risk logic)
                dynamic_qty = calculate_dynamic_qty(
                    confidence,
                    effective_H, 
                    effective_V,  
                    current_capital, 
                    curr_close,
                    sl,               # Pass the calculated stop loss
                    is_forex
                )
                
                # Guard clause
                if dynamic_qty == 0:
                    continue 
                
                # 3. Finalize TP based on conviction
                tp_dist = abs(tp_base - curr_close) * (1.0 + 0.5 * confidence)
                final_tp = curr_close + (tp_dist * effective_dir)

                active_position = Position(
                    ticker = quote_name,
                    entry_index = i,
                    entry_price =curr_close,
                    entry_force = f_val,
                    side = effective_dir,
                    quantity = dynamic_qty,
                    take_profit = float(final_tp),
                    stop_loss = float(sl),
                    state = []
                )

                commission = dynamic_qty * 0.005
                current_capital -= commission   
        # --- 3. EQUITY TRACKING ---
        unrealized = 0.0
        if active_position:
            unrealized = float((curr_close - active_position.entry_price) * active_position.side * active_position.quantity)
            active_position.state.append(
                State(
                    index = i,
                    open_price = float(row['OPEN']),
                    high_price = float(row['HIGH']),
                    low_price = float(row['LOW']),
                    close_price = curr_close,
                    δP = curr_dp,
                    V = effective_V,
                    H = H
                )
            )

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
                
                if not (len (X_volume_time_test) == len(X_price_time_test) and len(X_volume_time_test) == len(X_force_test)):
                    X_force_test = X_force_test.reindex(X_volume_time_test.index)
                    print(f"Data length mismatch for {quote_name}: Price-Time={len(X_price_time_test)}, Volume-Time={len(X_volume_time_test)}, Force={len(X_force_test)}. Skipping.")
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
                            "position_history": [{"index": s.index, "open_price": float(s.open_price), "high_price": float(s.high_price), "low_price": float(s.low_price), "close_price": float(s.close_price), "dP": float(s.δP), "V": float(s.V), "H": float(s.H)}  for s in transaction.state]
                        }
                        transaction_list.append(transaction)
                    print(json.dumps(transaction_list), file=f)
            else:
                print(f"Skipping {quote_name} (Not Tradable or No Data)")
        connection.close()
    engine.dispose()