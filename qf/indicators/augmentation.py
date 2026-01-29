def add_volatility_columns(df, lookback_periods):
    direction = abs(df['CLOSE'] - df['CLOSE'].shift(lookback_periods))
    volatility = abs(df['CLOSE'] - df['CLOSE'].shift(1)).rolling(window=lookback_periods).sum()
    df['ER'] = direction / (volatility + 1e-9)
    df['ATR %'] = (volatility / lookback_periods) / df['CLOSE']
    df.dropna(inplace=True)
