        
CREATE OR REPLACE FUNCTION "ω" ("Θ1" DOUBLE precision, "Θ2" DOUBLE precision, "Θ3" DOUBLE precision, "Θ4" DOUBLE PRECISION)
returns DOUBLE precision
LANGUAGE plpgsql AS $$
BEGIN
	RETURN (COS("Θ1") + SIN("Θ1") + COS("Θ2") + SIN("Θ2") + COS("Θ3") + SIN("Θ3") + COS("Θ4") + SIN("Θ4")) / (4 * SQRT(2));
END;
$$;


CREATE OR REPLACE FUNCTION "Ω" ("Θ1" DOUBLE precision, "Θ2" DOUBLE precision, "Θ3" DOUBLE precision, "Θ4" DOUBLE PRECISION)
returns DOUBLE precision
LANGUAGE plpgsql AS $$
BEGIN
	 return (128/(power(pi(),4) + 2*power(pi(),3) + 48*power(pi(),2))) * power("ω"("Θ1", "Θ2", "Θ3", "Θ4"),2);
END;
$$;

CREATE OR REPLACE FUNCTION "h" ("Θ1" DOUBLE precision, "Θ2" DOUBLE precision)
returns DOUBLE precision
LANGUAGE plpgsql AS $$
BEGIN
	RETURN (COS("Θ1") + SIN("Θ1") + COS("Θ2") + SIN("Θ2")) / (2*sqrt(2));
END;
$$;

CREATE OR REPLACE FUNCTION "H" ("Θ1" DOUBLE precision, "Θ2" DOUBLE precision)
returns DOUBLE precision
LANGUAGE plpgsql AS $$
BEGIN
	 return  (16/(power(pi(),2) + 2*pi() + 16)) * power("h"("Θ1", "Θ2"),2);
END;
$$;

DROP FUNCTION IF EXISTS ANGULAR_INDICATORS(TICKER VARCHAR(20));

CREATE OR REPLACE FUNCTION ANGULAR_INDICATORS(TICKER_NAME VARCHAR(20))
RETURNS TABLE (
	ticker VARCHAR(20), 
	quote_timestamp TIMESTAMP,
	"LP" DOUBLE PRECISION,       
	"B" DOUBLE PRECISION,
	"Ω" DOUBLE PRECISION,
	"H" DOUBLE PRECISION,
	"f" DOUBLE PRECISION
)
LANGUAGE plpgsql AS $$
BEGIN
     RETURN QUERY WITH BARS AS (
        SELECT Q.ticker,
		Q.quote_timestamp, 
        LN(Q.close_price / LAG(Q.close_price, 1) OVER (PARTITION BY Q.ticker ORDER BY Q.quote_timestamp)) "LP",
        (Q.close_price - Q.open_price) / CASE WHEN (Q.high_price - Q.low_price) = 0 THEN 1 ELSE (Q.high_price - Q.low_price) END  "B",		
        Q.high_price,
        Q.low_price,
        Q.close_price,
        Q.volume,
        ROW_NUMBER() OVER (PARTITION BY Q.ticker ORDER BY Q.quote_timestamp) as rn
        FROM quote_stocks Q
		WHERE Q.ticker = TICKER_NAME AND (Q.open_price > 0 AND Q.high_price > 0 AND Q.low_price > 0 AND Q.close_price > 0)
        ORDER BY Q.ticker, Q.quote_timestamp
    ),
    price_pivots AS (
        SELECT 
            curr.rn, curr.ticker, curr.quote_timestamp, curr.close_price, curr.high_price, curr.low_price, curr.volume,            
	        curr."LP",    
            (curr.rn - h_up.rn) AS i_h_up, (h_up.high_price - curr.high_price) AS p_h_up,
            (curr.rn - h_dn.rn) AS i_h_dn, (h_dn.high_price - curr.high_price) AS p_h_dn,
            (curr.rn - l_up.rn) AS i_l_up, (l_up.low_price - curr.low_price) AS p_l_up,
            (curr.rn - l_dn.rn) AS i_l_dn, (l_dn.low_price - curr.low_price) AS p_l_dn,
			curr."B"         			
        FROM BARS curr
        -- 3. Add Ticker Correlation to Lateral Joins
        LEFT JOIN LATERAL (SELECT rn, high_price FROM bars prev WHERE prev.ticker = curr.ticker AND prev.rn < curr.rn AND prev.high_price > curr.high_price ORDER BY prev.rn DESC LIMIT 1) h_up ON TRUE
        LEFT JOIN LATERAL (SELECT rn, high_price FROM bars prev WHERE prev.ticker = curr.ticker AND prev.rn < curr.rn AND prev.high_price < curr.high_price ORDER BY prev.rn DESC LIMIT 1) h_dn ON TRUE
        LEFT JOIN LATERAL (SELECT rn, low_price  FROM bars prev WHERE prev.ticker = curr.ticker AND prev.rn < curr.rn AND prev.low_price  > curr.low_price  ORDER BY prev.rn DESC LIMIT 1) l_up ON TRUE
        LEFT JOIN LATERAL (SELECT rn, low_price  FROM bars prev WHERE prev.ticker = curr.ticker AND prev.rn < curr.rn AND prev.low_price  < curr.low_price  ORDER BY prev.rn DESC LIMIT 1) l_dn ON TRUE
    ),
    price_bases AS (
        SELECT *, GREATEST(i_h_up, i_h_dn, i_l_up, i_l_dn) AS B_t, GREATEST(ABS(p_h_up), ABS(p_h_dn), ABS(p_l_up), ABS(p_l_dn)) AS C_t
        FROM price_pivots
    ),
    price_angles as (
     	SELECT 
     	   rn, price_bases.ticker, price_bases.quote_timestamp, 
	       price_bases."LP",
		   price_bases.volume,
	       price_bases."B",
	       COALESCE(ATAN((1.0 * i_h_up / NULLIF(B_t, 0)) / ((1.0 * p_h_up / NULLIF(C_t, 0)) + 0.000009)), 0) AS "Θh↑",
	       COALESCE(ATAN((1.0 * i_h_dn / NULLIF(B_t, 0)) / ((1.0 * p_h_dn / NULLIF(C_t, 0)) + 0.000009)), 0) AS "Θh↓",
	       COALESCE(ATAN((1.0 * i_l_up / NULLIF(B_t, 0)) / ((1.0 * p_l_up / NULLIF(C_t, 0)) + 0.000009)), 0) AS "Θl↑",
	       COALESCE(ATAN((1.0 * i_l_dn / NULLIF(B_t, 0)) / ((1.0 * p_l_dn / NULLIF(C_t, 0)) + 0.000009)), 0) AS "Θl↓"
		FROM price_bases
    ),
    volume_pivots as (SELECT 
            curr.ticker,
			curr.quote_timestamp,
            curr."LP", 
            curr."Θh↑", 
			curr."Θh↓", 
			curr."Θl↑",
			curr."Θl↓",	 
            curr."B",
            (curr.rn - v_up.rn) AS i_v_up, (v_up.volume - curr.volume) AS val_v_up,
            (curr.rn - v_dn.rn) AS i_v_dn, (v_dn.volume - curr.volume) AS val_v_dn
        FROM price_angles curr
        -- 4. Add Ticker Correlation to Volume Lateral Joins
        LEFT JOIN LATERAL (
            SELECT rn, volume FROM price_angles 
            WHERE price_angles.ticker = curr.ticker AND rn < curr.rn AND volume > curr.volume 
            ORDER BY rn DESC LIMIT 1
        ) v_up ON TRUE
        LEFT JOIN LATERAL (
            SELECT rn, volume FROM price_angles 
            WHERE price_angles.ticker = curr.ticker AND rn < curr.rn AND volume < curr.volume 
            ORDER BY rn DESC LIMIT 1
        ) v_dn ON TRUE
     ),
     volume_bases as (
        SELECT 
            *,           
            GREATEST(i_v_up, i_v_dn, 1) AS Vol_B_t,
            GREATEST(ABS(val_v_up), ABS(val_v_dn), 0.000009) AS Vol_C_t
        FROM volume_pivots  
     ), all_angles as (
	     SELECT             
	        volume_bases.ticker,
			volume_bases.quote_timestamp,
	        volume_bases."LP",       
	        volume_bases."Θh↑",
			volume_bases."Θh↓",
			volume_bases."Θl↑",
			volume_bases."Θl↓",	
	        ATAN(COALESCE((1.0 * i_v_up / NULLIF(Vol_B_t, 0)) / ((1.0 * val_v_up / NULLIF(Vol_C_t, 0)) + 0.000009), 0)) AS "φ1",
	        ATAN(COALESCE((1.0 * i_v_dn / NULLIF(Vol_B_t, 0)) / ((1.0 * val_v_dn / NULLIF(Vol_C_t, 0)) + 0.000009), 0)) AS "φ2",
	        volume_bases."B"
	    FROM volume_bases
	), wavelets AS (
		 SELECT             
		    a.ticker,
			a.quote_timestamp,
		    a."LP",       
		    a."B",
	        "Ω"(a."Θh↑", a."Θh↓", a."Θl↑", a."Θl↓") as "Ω",
			"H"("φ1", "φ2") as "H",
            "F"(a."Θh↑", a."Θh↓", a."Θl↑", a."Θl↓", a."φ1", a."φ2") as "F"
		FROM all_angles a
	)
	SELECT
	    w.ticker,
		w.quote_timestamp,
	    w."LP"::DOUBLE PRECISION,       
	    w."B"::DOUBLE PRECISION,
        w."Ω"::DOUBLE PRECISION,
		w."H"::DOUBLE PRECISION,
        w."F" as "f"
	FROM wavelets w
    ORDER BY w.ticker, w.quote_timestamp;
END;
$$;