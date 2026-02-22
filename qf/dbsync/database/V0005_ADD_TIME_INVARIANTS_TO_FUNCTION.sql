DROP FUNCTION IF EXISTS ANGULAR_INDICATORS(TICKER VARCHAR(20));
CREATE OR REPLACE FUNCTION ANGULAR_INDICATORS(TICKER VARCHAR(20))
RETURNS TABLE (
	"TICKER" VARCHAR(20), 
	"TIMESTAMP" TIMESTAMP,
	"ΔP" DOUBLE PRECISION,       
	"B" DOUBLE PRECISION,
	"Ω" DOUBLE PRECISION,
	"H" DOUBLE PRECISION,
	"Ω⋅ΔP" DOUBLE PRECISION,
	"H⋅ΔP" DOUBLE PRECISION,
    "MARKET_CAP_LOG10" DOUBLE PRECISION,
    "BETA_LOG" DOUBLE PRECISION
)
LANGUAGE plpgsql AS $$
BEGIN
     RETURN QUERY WITH BARS AS (
        SELECT Q."TICKER",
		Q."TIMESTAMP", 
        (Q."CLOSE" - LAG(Q."CLOSE", 1) OVER (PARTITION BY Q."TICKER" ORDER BY Q."TIMESTAMP")) / (Q."CLOSE" + LAG(Q."CLOSE", 1) OVER (PARTITION BY Q."TICKER" ORDER BY Q."TIMESTAMP")) "ΔP",
        (Q."CLOSE" - Q."OPEN") / CASE WHEN (Q."HIGH" - Q."LOW") = 0 THEN 1 ELSE (Q."HIGH" - Q."LOW") END  "B",		
        Q."HIGH",
        Q."LOW",
        Q."CLOSE",
        Q."VOLUME",
        ROW_NUMBER() OVER (PARTITION BY Q."TICKER" ORDER BY Q."TIMESTAMP") as rn
        FROM "QUOTE" Q
		WHERE Q."TICKER" = TICKER AND (Q."OPEN" > 0 AND Q."HIGH" > 0 AND Q."LOW" > 0 AND Q."CLOSE" > 0)
        ORDER BY Q."TICKER", Q."TIMESTAMP"
    ),
    price_pivots AS (
        SELECT 
            curr.rn, curr."TICKER", curr."TIMESTAMP", curr."CLOSE", curr."HIGH", curr."LOW", curr."VOLUME",            
	        curr."ΔP",    
            (curr.rn - h_up.rn) AS i_h_up, (h_up."HIGH" - curr."HIGH") AS p_h_up,
            (curr.rn - h_dn.rn) AS i_h_dn, (curr."HIGH" - h_dn."HIGH") AS p_h_dn,
            (curr.rn - l_up.rn) AS i_l_up, (l_up."LOW" - curr."LOW") AS p_l_up,
            (curr.rn - l_dn.rn) AS i_l_dn, (curr."LOW" - l_dn."LOW") AS p_l_dn,
			curr."B"         			
        FROM BARS curr
        -- 3. Add Ticker Correlation to Lateral Joins
        LEFT JOIN LATERAL (SELECT rn, "HIGH" FROM bars prev WHERE prev."TICKER" = curr."TICKER" AND prev.rn < curr.rn AND prev."HIGH" > curr."HIGH" ORDER BY prev.rn DESC LIMIT 1) h_up ON TRUE
        LEFT JOIN LATERAL (SELECT rn, "HIGH" FROM bars prev WHERE prev."TICKER" = curr."TICKER" AND prev.rn < curr.rn AND prev."HIGH" < curr."HIGH" ORDER BY prev.rn DESC LIMIT 1) h_dn ON TRUE
        LEFT JOIN LATERAL (SELECT rn, "LOW"  FROM bars prev WHERE prev."TICKER" = curr."TICKER" AND prev.rn < curr.rn AND prev."LOW"  > curr."LOW"  ORDER BY prev.rn DESC LIMIT 1) l_up ON TRUE
        LEFT JOIN LATERAL (SELECT rn, "LOW"  FROM bars prev WHERE prev."TICKER" = curr."TICKER" AND prev.rn < curr.rn AND prev."LOW"  < curr."LOW"  ORDER BY prev.rn DESC LIMIT 1) l_dn ON TRUE
    ),
    price_bases AS (
        SELECT *, GREATEST(i_h_up, i_h_dn, i_l_up, i_l_dn) AS B_t, GREATEST(p_h_up, p_h_dn, p_l_up, p_l_dn) AS C_t
        FROM price_pivots
    ),
    price_angles as (
     	SELECT 
     	   rn, price_bases."TICKER", price_bases."TIMESTAMP", 
	       price_bases."ΔP",
		   price_bases."VOLUME",
	       price_bases."B",
	       COALESCE(ATAN((1.0 * i_h_up / NULLIF(B_t, 0)) / ((1.0 * p_h_up / NULLIF(C_t, 0)) + 0.000009)), 0) AS "Θh↑",
	       COALESCE(ATAN((1.0 * i_h_dn / NULLIF(B_t, 0)) / ((1.0 * p_h_dn / NULLIF(C_t, 0)) + 0.000009)), 0) AS "Θh↓",
	       COALESCE(ATAN((1.0 * i_l_up / NULLIF(B_t, 0)) / ((1.0 * p_l_up / NULLIF(C_t, 0)) + 0.000009)), 0) AS "Θl↑",
	       COALESCE(ATAN((1.0 * i_l_dn / NULLIF(B_t, 0)) / ((1.0 * p_l_dn / NULLIF(C_t, 0)) + 0.000009)), 0) AS "Θl↓"
		FROM price_bases
    ),
    volume_pivots as (SELECT 
            curr."TICKER",
			curr."TIMESTAMP",
            curr."ΔP", 
            curr."Θh↑", 
			curr."Θh↓", 
			curr."Θl↑",
			curr."Θl↓",	 
            curr."B",
            (curr.rn - v_up.rn) AS i_v_up, (v_up."VOLUME" - curr."VOLUME") AS val_v_up,
            (curr.rn - v_dn.rn) AS i_v_dn, (curr."VOLUME" - v_dn."VOLUME") AS val_v_dn
        FROM price_angles curr
        -- 4. Add Ticker Correlation to Volume Lateral Joins
        LEFT JOIN LATERAL (
            SELECT rn, "VOLUME" FROM price_angles 
            WHERE price_angles."TICKER" = curr."TICKER" AND rn < curr.rn AND "VOLUME" > curr."VOLUME" 
            ORDER BY rn DESC LIMIT 1
        ) v_up ON TRUE
        LEFT JOIN LATERAL (
            SELECT rn, "VOLUME" FROM price_angles 
            WHERE price_angles."TICKER" = curr."TICKER" AND rn < curr.rn AND "VOLUME" < curr."VOLUME" 
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
	        volume_bases."TICKER",
			volume_bases."TIMESTAMP",
	        volume_bases."ΔP",       
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
		    a."TICKER",
			a."TIMESTAMP",
		    a."ΔP",       
		    a."B",
	        "Ω"(a."Θh↑", a."Θh↓", a."Θl↑", a."Θl↓") as "Ω",
			"H"("φ1", "φ2") as "H"
		FROM all_angles a
	)
	SELECT
	    w."TICKER",
		w."TIMESTAMP",
	    w."ΔP"::DOUBLE PRECISION,       
	    w."B"::DOUBLE PRECISION,
        w."Ω"::DOUBLE PRECISION,
		w."H"::DOUBLE PRECISION,
	    (w."Ω" * w."ΔP"  * 100)::DOUBLE PRECISION AS "Ω⋅ΔP",
		(w."H" * w."ΔP" * 100)::DOUBLE PRECISION AS "H⋅ΔP",
        q."MARKET_CAP_LOG10"::DOUBLE PRECISION,
        q."BETA_LOG"::DOUBLE PRECISION
	FROM wavelets w inner join "QUOTE" q on w."TICKER" = q."TICKER" and w."TIMESTAMP" = q."TIMESTAMP"
    ORDER BY w."TICKER", w."TIMESTAMP";
END;
$$;