from qf.indicators.augmentation import add_volatility_columns
from sqlalchemy import create_engine
import pandas as pd
#  (vol_wavelet_momentum - LAG(vol_wavelet_momentum, 1) OVER (ORDER BY "TIMESTAMP")) as v_target
def volumetime_sql(quote_name, lookback_periods):
    lags = []
    features = set()
    scale_factor = 10

    # Feature generation: Using double % to escape for SQLAlchemy if needed, 
    # but here we use a clean 'pct' naming convention.
    for i in range(1, lookback_periods + 1): 
        term = f'''(LAG(v_mom, {i}) OVER (order by "TIMESTAMP") - LAG(v_mom, {i+1}) OVER (order by "TIMESTAMP")) * {scale_factor} v{i}'''
        lags.append(term)
        features.add(f"v{i}")
    
    # Using 'v_target' as a safe ASCII alias for the Python side
    return (f"""		
    WITH quote_indexed AS (
        SELECT "TICKER", "TIMESTAMP", "VOLUME", "CLOSE",
               ROW_NUMBER() OVER (ORDER BY "TIMESTAMP") as rn
        FROM QUOTE
        WHERE "TICKER" = '{quote_name}' 
        ORDER BY "TIMESTAMP"
    ),
    structural_pivots AS (
        SELECT 
            curr.rn, curr."TICKER", curr."TIMESTAMP", curr."VOLUME", curr."CLOSE",
            (curr.rn - v_up.rn) AS i_v_up, 
            (v_up."VOLUME" - curr."VOLUME") AS val_v_up,
            (curr.rn - v_dn.rn) AS i_v_dn, 
            (curr."VOLUME" - v_dn."VOLUME") AS val_v_dn
        FROM quote_indexed curr
        LEFT JOIN LATERAL (
            SELECT rn, "VOLUME" FROM quote_indexed 
            WHERE rn < curr.rn AND "VOLUME" > curr."VOLUME" 
            ORDER BY rn DESC LIMIT 1
        ) v_up ON TRUE
        LEFT JOIN LATERAL (
            SELECT rn, "VOLUME" FROM quote_indexed 
            WHERE rn < curr.rn AND "VOLUME" < curr."VOLUME" 
            ORDER BY rn DESC LIMIT 1
        ) v_dn ON TRUE
        ORDER BY curr."TIMESTAMP"
    ),
    bases AS (
        SELECT *,
            GREATEST(i_v_up, i_v_dn, 1) AS B_t,
            GREATEST(ABS(val_v_up), ABS(val_v_dn), 0.000009) AS C_t
        FROM structural_pivots
        ORDER BY "TIMESTAMP"
    ),
    angles AS (
        SELECT *,
            ATAN(COALESCE((1.0 * i_v_up / NULLIF(B_t, 0)) / ((1.0 * val_v_up / NULLIF(C_t, 0)) + 0.000009), 0)) AS "φ1",
            ATAN(COALESCE((1.0 * i_v_dn / NULLIF(B_t, 0)) / ((1.0 * val_v_dn / NULLIF(C_t, 0)) + 0.000009), 0)) AS "φ2"
        FROM bases
        ORDER BY "TIMESTAMP"
    ),
    base_momentum AS (
        SELECT 
            "TIMESTAMP", "CLOSE", angles."φ1", angles."φ2",
            (
                (POWER(COS("φ1") + SIN("φ1"), 2) + POWER(COS("φ1") + SIN("φ1"), 2)) / 4) * (
                ("CLOSE" - LAG("CLOSE", 1) OVER (order by "TIMESTAMP")) / (ABS("CLOSE") + ABS(LAG("CLOSE", 1) OVER (order by "TIMESTAMP")) + 0.000009)
            ) AS v_mom,
            (POWER(COS("φ1") + SIN("φ1"), 2) + POWER(COS("φ1") + SIN("φ1"), 2)) / 4 "v(t)"
        FROM angles
        ORDER BY "TIMESTAMP"
    )
    SELECT "TIMESTAMP", v_mom, "CLOSE", base_momentum."φ1", base_momentum."φ2", "v(t)",
    (v_mom - LAG(v_mom, 1) OVER (ORDER BY "TIMESTAMP")) * {scale_factor}  AS v_target,
    {", ".join(lags)}    
    FROM base_momentum
    ORDER BY "TIMESTAMP"
    """, list(sorted(features)), "v_target")
    
def volumetime(sqlalchemy_url, quote_name, lookback_periods):
    (sql_template, features, target) = volumetime_sql(quote_name, lookback_periods)
    
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