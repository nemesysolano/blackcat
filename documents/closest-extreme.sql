   WITH quote_indexed AS (
        SELECT "TICKER", "TIMESTAMP", "VOLUME", "CLOSE",
               ROW_NUMBER() OVER (ORDER BY "TIMESTAMP") as rn
        FROM QUOTE
        WHERE "TICKER" = 'AAPL' 
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
    ),
    bases AS (
        SELECT *,
            GREATEST(i_v_up, i_v_dn, 1) AS B_t,
            GREATEST(ABS(val_v_up), ABS(val_v_dn), 0.000009) AS C_t
        FROM structural_pivots
    ),
    angles AS (
        SELECT *,
            ATAN(COALESCE((1.0 * i_v_up / NULLIF(B_t, 0)) / ((1.0 * val_v_up / NULLIF(C_t, 0)) + 0.000009), 0)) AS phi1,
            ATAN(COALESCE((1.0 * i_v_dn / NULLIF(B_t, 0)) / ((1.0 * val_v_dn / NULLIF(C_t, 0)) + 0.000009), 0)) AS phi2
        FROM bases
    ),
    base_momentum AS (
        SELECT 
            "TIMESTAMP", "CLOSE",
            (
                (POWER(COS(phi1) + SIN(phi1), 2) + POWER(COS(phi2) + SIN(phi2), 2)) / 4) * (
                ("CLOSE" - LAG("CLOSE", 1) OVER (order by "TIMESTAMP")) / (ABS("CLOSE") + ABS(LAG("CLOSE", 1) OVER (order by "TIMESTAMP")) + 0.000009)
            ) AS v_mom
        FROM angles
    ),
    SELECT "TIMESTAMP",
    (v_mom - LAG(v_mom, 1) OVER (ORDER BY "TIMESTAMP"))  AS v_target,
    (LAG(v_mom, 1) OVER (order by "TIMESTAMP") - LAG(v_mom, 2) OVER (order by "TIMESTAMP")) v1, (LAG(v_mom, 2) OVER (order by "TIMESTAMP") - LAG(v_mom, 3) OVER (order by "TIMESTAMP")) v2, (LAG(v_mom, 3) OVER (order by "TIMESTAMP") - LAG(v_mom, 4) OVER (order by "TIMESTAMP")) v3, (LAG(v_mom, 4) OVER (order by "TIMESTAMP") - LAG(v_mom, 5) OVER (order by "TIMESTAMP")) v4, (LAG(v_mom, 5) OVER (order by "TIMESTAMP") - LAG(v_mom, 6) OVER (order by "TIMESTAMP")) v5, (LAG(v_mom, 6) OVER (order by "TIMESTAMP") - LAG(v_mom, 7) OVER (order by "TIMESTAMP")) v6, (LAG(v_mom, 7) OVER (order by "TIMESTAMP") - LAG(v_mom, 8) OVER (order by "TIMESTAMP")) v7, (LAG(v_mom, 8) OVER (order by "TIMESTAMP") - LAG(v_mom, 9) OVER (order by "TIMESTAMP")) v8, (LAG(v_mom, 9) OVER (order by "TIMESTAMP") - LAG(v_mom, 10) OVER (order by "TIMESTAMP")) v9, (LAG(v_mom, 10) OVER (order by "TIMESTAMP") - LAG(v_mom, 11) OVER (order by "TIMESTAMP")) v10, (LAG(v_mom, 11) OVER (order by "TIMESTAMP") - LAG(v_mom, 12) OVER (order by "TIMESTAMP")) v11, (LAG(v_mom, 12) OVER (order by "TIMESTAMP") - LAG(v_mom, 13) OVER (order by "TIMESTAMP")) v12, (LAG(v_mom, 13) OVER (order by "TIMESTAMP") - LAG(v_mom, 14) OVER (order by "TIMESTAMP")) v13, (LAG(v_mom, 14) OVER (order by "TIMESTAMP") - LAG(v_mom, 15) OVER (order by "TIMESTAMP")) v14    
    FROM base_momentum
    ORDER BY "TIMESTAMP"