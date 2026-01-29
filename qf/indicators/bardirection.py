from qf.indicators.augmentation import add_volatility_columns
from sqlalchemy import create_engine
import pandas as pd

def bardirection_sql(quote_name, lookback_periods):
    lags = []
    features = set()
    scale_factor = 0.025
    
    # Range is 1 to lookback_periods (inclusive)
    # i=1 generates Feature Y1: LAG(1) - LAG(2) (No Leak)
    for i in range(1, lookback_periods+1): 
        term = f'(LAG(d, {i}) OVER (order by "TIMESTAMP") - LAG(d, {i+1}) OVER (order by "TIMESTAMP")) * {scale_factor} d{i}'
        lags.append(term)
        features.add(f"d{i}")

    return ((
    f"""		
    with direction as (
        select "TICKER", "TIMESTAMP", "CLOSE", case when ABS("CLOSE" - "OPEN")  > 0.00000001 THEN  (("CLOSE" - "OPEN")/("HIGH" - "LOW" + 0.000001)) else 0  end d from quote
        where "TICKER" = '{quote_name}'
        order by "TIMESTAMP"
    )
    select "TICKER", "TIMESTAMP", "CLOSE", d,
    (d - LAG(d, 1) OVER (order by "TIMESTAMP"))  * {scale_factor} AS d_target,
    {",".join(lags)}    
    from direction
    order by "TIMESTAMP"
    """
    ), list(sorted(features)), "d_target")

def bardirection(sqlalchemy_url, quote_name, lookback_periods):
    (sql_template, features, target) = bardirection_sql(quote_name, lookback_periods)
    engine = create_engine(sqlalchemy_url)
    
    with engine.connect() as connection:
        df = pd.read_sql(sql_template, connection)
        # engine.connect() used in a 'with' block handles closing, 
        # but explicit close is fine.
        connection.close()
    engine.dispose()
    
    # Drop rows where LAG functions returned NULL (the beginning of the dataset)
    df.dropna(inplace=True)
    df.set_index('TIMESTAMP', inplace=True)
    add_volatility_columns(df, lookback_periods)
    return df, features, target