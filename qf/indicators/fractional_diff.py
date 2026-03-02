import pandas as pd
def fractional_price_acceleration_sql (quote_name, lookback_periods):
    wavelet_lags = []
    features = []    
    wavelet_lags.append(f"LOG(\"CLOSE\" / LAG(X.\"CLOSE\", 1) OVER(ORDER BY \"TIMESTAMP\")) \"L1\"")
    features.append(f"L1")

    for i in range(2, lookback_periods + 1):
        wavelet_lags.append(f"LOG(LAG(\"CLOSE\", {i}) OVER(ORDER BY \"TIMESTAMP\") / LAG(\"CLOSE\", {i+1}) OVER(ORDER BY \"TIMESTAMP\")) \"L{i}\"")
        features.append(f"L{i}")

    wavelet_lags.reverse()  # Reverse to have the most recent lag first
    features.reverse()  # Reverse to match the order of the lags
    lags_str = ", ".join(wavelet_lags)
    return (f"SELECT \"TIMESTAMP\", {lags_str}, LOG(\"CLOSE\" /  LAG(\"CLOSE\", 1) OVER(ORDER BY \"TIMESTAMP\")) - LOG(LAG(\"CLOSE\", 1) OVER(ORDER BY \"TIMESTAMP\") / LAG(\"CLOSE\", 2) OVER(ORDER BY \"TIMESTAMP\")) AS \"A\" FROM \"QUOTE\" X WHERE \"TICKER\" = '{quote_name}'", features, "A")


def fractional_price_acceleration(connection, quote_name, lookback_periods):
    (sql_template, features, target) = fractional_price_acceleration_sql(quote_name, lookback_periods)
    df = pd.read_sql(sql_template, connection)
    df.dropna(inplace=True)
    df.set_index('TIMESTAMP', inplace=True)
    return df, features, target
