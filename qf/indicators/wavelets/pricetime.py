from qf.indicators.augmentation import add_volatility_columns
from sqlalchemy import create_engine
import pandas as pd

def pricetime_sql(quote_name, lookback_periods):
    lags = []
    features = set()
    scale_factor = 1
    
    # Range is 1 to lookback_periods (inclusive)
    # i=1 generates Feature Y1: LAG(1) - LAG(2) (No Leak)
    for i in range(1, lookback_periods+1): 
        term = f'(LAG(W, {i}) OVER (order by "TIMESTAMP") - LAG(W, {i+1}) OVER (order by "TIMESTAMP")) * {scale_factor} w{i}'
        lags.append(term)
        features.add(f"w{i}")
    
    return ((
    f"""		
    WITH quote_indexed AS (
        SELECT  "TICKER", "TIMESTAMP", "CLOSE",  "HIGH",  "LOW",  ROW_NUMBER() OVER (ORDER BY "TIMESTAMP") as rn
        FROM QUOTE
        WHERE "TICKER" = '{quote_name}' 
    ),
    structural_pivots AS (
        SELECT 
            curr.rn, 
            curr."TICKER", 
            curr."TIMESTAMP", 
            curr."CLOSE", 
            curr."HIGH", 
            curr."LOW",
            (curr.rn - h_up.rn) AS i_h_up, 
            (h_up."HIGH" - curr."HIGH") AS p_h_up,
            (curr.rn - h_dn.rn) AS i_h_dn, 
            (curr."HIGH" - h_dn."HIGH") AS p_h_dn,
            (curr.rn - l_up.rn) AS i_l_up, 
            (l_up."LOW" - curr."LOW") AS p_l_up,
            (curr.rn - l_dn.rn) AS i_l_dn, 
            (curr."LOW" - l_dn."LOW") AS p_l_dn
        FROM quote_indexed curr
        LEFT JOIN LATERAL (SELECT rn, "HIGH" FROM quote_indexed prev WHERE prev.rn < curr.rn AND prev."HIGH" > curr."HIGH" ORDER BY prev.rn DESC LIMIT 1) h_up ON TRUE
        LEFT JOIN LATERAL (SELECT rn, "HIGH" FROM quote_indexed prev WHERE prev.rn < curr.rn AND prev."HIGH" < curr."HIGH" ORDER BY prev.rn DESC LIMIT 1) h_dn ON TRUE
        LEFT JOIN LATERAL (SELECT rn, "LOW"  FROM quote_indexed prev WHERE prev.rn < curr.rn AND prev."LOW"  > curr."LOW"  ORDER BY prev.rn DESC LIMIT 1) l_up ON TRUE
        LEFT JOIN LATERAL (SELECT rn, "LOW"  FROM quote_indexed prev WHERE prev.rn < curr.rn AND prev."LOW"  < curr."LOW"  ORDER BY prev.rn DESC LIMIT 1) l_dn ON TRUE
    ),
    bases AS (
        SELECT *, GREATEST(i_h_up, i_h_dn, i_l_up, i_l_dn) AS B_t, GREATEST(p_h_up, p_h_dn, p_l_up, p_l_dn) AS C_t
        FROM structural_pivots
    ), angles as (
        SELECT 
            "TICKER", "TIMESTAMP", "CLOSE",
            -- COALESCE ensures that if a pivot hasn't occurred yet, the angle defaults to 0
            COALESCE(ATAN((1.0 * i_h_up / NULLIF(B_t, 0)) / ((1.0 * p_h_up / NULLIF(C_t, 0)) + 0.000009)), 0) AS "Θh↑",
            COALESCE(ATAN((1.0 * i_h_dn / NULLIF(B_t, 0)) / ((1.0 * p_h_dn / NULLIF(C_t, 0)) + 0.000009)), 0) AS "Θh↓",
            COALESCE(ATAN((1.0 * i_l_up / NULLIF(B_t, 0)) / ((1.0 * p_l_up / NULLIF(C_t, 0)) + 0.000009)), 0) AS "Θl↑",
            COALESCE(ATAN((1.0 * i_l_dn / NULLIF(B_t, 0)) / ((1.0 * p_l_dn / NULLIF(C_t, 0)) + 0.000009)), 0) AS "Θl↓"
        FROM bases
        ORDER BY "TIMESTAMP"
    ), pricetime as (select 
        quote."TICKER", quote."TIMESTAMP",
        "Θh↑", "Θh↓", "Θl↑", "Θl↓", angles."CLOSE",
        ((POWER(cos("Θh↑") + sin("Θh↑"),2) + POWER(cos("Θh↓") + sin("Θh↓"),2) + POWER(cos("Θl↑") + sin("Θl↑"),2) + POWER(cos("Θl↓") + sin("Θl↓"),2)) / 8) * (
        (angles."CLOSE" - LAG(angles."CLOSE", 1) OVER (order by angles."TIMESTAMP")) / (ABS(angles."CLOSE") + ABS(LAG(angles."CLOSE", 1) OVER (order by angles."TIMESTAMP")) + 0.00009)
        ) as W,
        ((POWER(cos("Θh↑") + sin("Θh↑"),2) + POWER(cos("Θh↓") + sin("Θh↓"),2) + POWER(cos("Θl↑") + sin("Θl↑"),2) + POWER(cos("Θl↓") + sin("Θl↓"),2)) / 8) "w(t)"
        from ANGLES inner join quote on angles."TICKER" = quote."TICKER" and angles."TIMESTAMP" = quote."TIMESTAMP"
        order by angles."TIMESTAMP"
    ) select "TIMESTAMP", W, "w(t)", "CLOSE",
        (W - LAG(W, 1) OVER (order by "TIMESTAMP")) * {scale_factor}  AS w_target,
        "Θh↑", "Θh↓", "Θl↑", "Θl↓",  -- Add these to the final selection
        {",".join(lags)}    
        from pricetime
        order by "TIMESTAMP"
    """
    ), list(sorted(features)), "w_target")

def pricetime(sqlalchemy_url, quote_name, lookback_periods):
    (sql_template, features, target) = pricetime_sql(quote_name, lookback_periods)
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