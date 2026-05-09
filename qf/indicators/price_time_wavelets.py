from sqlalchemy import create_engine
import pandas as pd
import qf.dbsync.cache as cache
import numpy as np
import qf.nativemath as nativemath

def add_lagged_feature(df, column_name, indicator_array, lookback_periods):
    # Wrap the raw numpy array in a Series aligned to the DataFrame's index
    base_series = pd.Series(indicator_array, index=df.index)
    features = []
    # 1. Insert the lags first, counting backwards (e.g., LP3, then LP2, then LP1)
    for i in range(lookback_periods, 0, -1):
        features.append(f"{column_name}{i}")
        df[features[-1]] = base_series.shift(i)
        
    # 2. Insert the current un-lagged column last (e.g., LP)
    df[column_name] = base_series
    return features


def log_acceleration_sql(quote_name, lookback_periods): 
    return f"SELECT quote_timestamp, open_price, high_price, low_price, close_price, volume FROM quote_stocks WHERE ticker = '{quote_name}' ORDER BY quote_timestamp ASC"

def log_acceleration_direction(connection, redis_connection, quote_name, lookback_periods):
    sql_template = log_acceleration_sql(quote_name, lookback_periods)
    df = cache.fetch(connection, redis_connection, sql_template)
    high_price = df['high_price'].values.astype(np.float64)
    low_price = df['low_price'].values.astype(np.float64)
    close_price = df['close_price'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)
        
    
    L, Lambda, f = nativemath.get_price_time_indicators(close_price, high_price, low_price, volume, lookback_periods)
    df['f'] = f
    df['L'] = L
    features = add_lagged_feature(df, 'Lambda', Lambda, lookback_periods)
    df.dropna(inplace=True)    
    return df, features, 'Lambda'
