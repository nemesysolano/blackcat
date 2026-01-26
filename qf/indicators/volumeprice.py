from sqlalchemy import create_engine
import pandas as pd

def volumeprice_sql(quote_name, lookback_periods):
    volume_term = '1.0' if quote_name.endswith('=X') else 'ABS("VOLUME_DIFF%%")'
    lags = []
    features = set()
    scale_factor = 10
    
    # Range is 1 to lookback_periods (inclusive)
    # i=1 generates Feature Y1: LAG(1) - LAG(2) (No Leak)
    for i in range(1, lookback_periods+1): 
        term = f'(LAG(y, {i}) OVER (order by "TIMESTAMP") - LAG(Y, {i+1}) OVER (order by "TIMESTAMP")) * {scale_factor}  y{i}'
        lags.append(term)
        features.add(f"y{i}")
    
    return ((
    f"""		
        with PCT_DIFF as (
            select 	"TIMESTAMP", 
            "CLOSE", 
            ("CLOSE" - LAG("CLOSE", 1) OVER (order by "TIMESTAMP"))   / (ABS("CLOSE") + ABS( LAG("CLOSE",   1) OVER (order by "TIMESTAMP")) + 0.000009) "CLOSE_DIFF%%",
            "VOLUME",
            ("VOLUME" - LAG("VOLUME", 1) OVER (order by "TIMESTAMP")) / (ABS("VOLUME") + ABS( LAG("VOLUME", 1) OVER (order by "TIMESTAMP")) + 0.000009) "VOLUME_DIFF%%"
            FROM QUOTE
            where "TICKER" = '{quote_name}'
            order by "TIMESTAMP"
        ),
        PRICE_VOL as (
            select "TIMESTAMP", "CLOSE_DIFF%%" * {volume_term} y
            from PCT_DIFF
            order by "TIMESTAMP"
        )
        select "TIMESTAMP", 
        (y - LAG(Y, 1) OVER (order by "TIMESTAMP")) * {scale_factor}  AS y_target,
        {','.join(lags)}
        from PRICE_VOL 
        order by "TIMESTAMP"
    """
    ), list(sorted(features)), "y_target")

def volumeprice(sqlalchemy_url, quote_name, lookback_periods):
    (sql_template, features, target) = volumeprice_sql(quote_name, lookback_periods)
    
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
    return df, features, target