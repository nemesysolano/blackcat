from ctypes.wintypes import DOUBLE
import os
from sqlalchemy import create_engine
import pandas as pd

def price_time_wavelet_direction_sql(quote_name, lookback_periods):
    wavelet_lags = []
    features = []
    
    for i in range(1, lookback_periods + 1):
        wavelet_lags.append(f"(LAG(X.\"Ω⋅ΔP\", {i}) OVER(ORDER BY \"TIMESTAMP\") - LAG(X.\"Ω⋅ΔP\", {i+1}) OVER(ORDER BY \"TIMESTAMP\")) AS \"Ω⋅ΔP{i}\"")
        features.append(f"Ω⋅ΔP{i}")

    wavelet_lags.reverse()  # Reverse to have the most recent lag first
    features.reverse()  # Reverse to match the order of the lags

    lags_str = ", ".join(wavelet_lags)
    return (f"SELECT \"TIMESTAMP\", {lags_str}, \"Ω⋅ΔP\" - LAG(\"Ω⋅ΔP\", 1) OVER(ORDER BY \"TIMESTAMP\") AS \"Ω⋅ΔP0\" FROM ANGULAR_INDICATORS('{quote_name}') X", features, "Ω⋅ΔP0") # Ω⋅ΔP

def price_time_wavelet_force_sql(quote_name, lookback_periods):
    wavelet_lags = []
    features = []
    for i in range(1, lookback_periods + 1):
        wavelet_lags.append(f"LAG(X.\"Ω\", {i}) OVER(ORDER BY \"TIMESTAMP\") AS \"Ω{i}\"")
        features.append(f"Ω{i}")

    wavelet_lags.reverse()  # Reverse to have the most recent lag first
    features.reverse()  # Reverse to match the order of the lags

    lags_str = ", ".join(wavelet_lags)
    return (f"SELECT \"TIMESTAMP\", {lags_str}, X.\"Ω\" FROM ANGULAR_INDICATORS('{quote_name}') X", features, "Ω")

def price_time_wavelet_direction(connection, quote_name, lookback_periods):
    (sql_template, features, target) = price_time_wavelet_direction_sql(quote_name, lookback_periods)
    
    df = pd.read_sql(sql_template, connection)    
    df.dropna(inplace=True)
    df.set_index('TIMESTAMP', inplace=True)
    return df, features, target

def price_time_wavelet_force(connection, quote_name, lookback_periods):
    (sql_template, features, target) = price_time_wavelet_force_sql(quote_name, lookback_periods)
    
    df = pd.read_sql(sql_template, connection)
    df.dropna(inplace=True)
    df.set_index('TIMESTAMP', inplace=True)
    return df, features, target
