		-- select * from ANGULAR_INDICATORS('AAPL')
create or replace function ANGULAR_INDICATORS(TICKER_NAME VARCHAR(20))
returns table (
	"TICKER" VARCHAR(20), 
	"TIMESTAMP" TIMESTAMP,
    "ΔP" DOUBLE PRECISION,
    "ΔV" DOUBLE PRECISION,
    "ΔH" DOUBLE PRECISION,
    "ΔH2" DOUBLE PRECISION,
    "ΔL" DOUBLE PRECISION,
    "ΔL2" DOUBLE PRECISION,	        
    "Θh↑" DOUBLE PRECISION,
    "Θh↓" DOUBLE PRECISION,
    "Θl↑" DOUBLE PRECISION,
    "Θl↓" DOUBLE PRECISION,	
    "φ1" DOUBLE PRECISION,
    "φ2" DOUBLE PRECISION,
    Bb DOUBLE PRECISION,
    Bf DOUBLE precision,
    Fb DOUBLE PRECISION,
    Ff DOUBLE PRECISION	 
)
LANGUAGE plpgsql AS $$
BEGIN
    -- This directive resolves the ambiguity by forcing PL/pgSQL to treat 
    -- ambiguous names (like "TICKER") as table columns, not output variables.
    
    RETURN QUERY WITH BARS AS (
        SELECT Q."TICKER", Q."TIMESTAMP", 
        LOG(LAG("CLOSE", 1) OVER (PARTITION BY Q."TICKER" ORDER BY Q."TIMESTAMP") / "CLOSE") * 100 "ΔP",
        LOG(LAG("VOLUME", 1) OVER (PARTITION BY Q."TICKER" ORDER BY Q."TIMESTAMP") / "VOLUME")*10 "ΔV",
        LOG(LAG("HIGH", 1) OVER (PARTITION BY Q."TICKER" ORDER BY Q."TIMESTAMP") / "HIGH")*100 "ΔH",
        LOG(LAG("HIGH", 2) OVER (PARTITION BY Q."TICKER" ORDER BY Q."TIMESTAMP") / "HIGH")*100 "ΔH2",
        LOG(LAG("LOW", 1) OVER (PARTITION BY Q."TICKER" ORDER BY Q."TIMESTAMP") / "LOW")*100 "ΔL",
        LOG(LAG("LOW", 2) OVER (PARTITION BY Q."TICKER" ORDER BY Q."TIMESTAMP") / "LOW")*100 "ΔL2",
        ("CLOSE" - LAG("CLOSE", 1) OVER (PARTITION BY Q."TICKER" ORDER BY Q."TIMESTAMP")) / (ABS("CLOSE") + ABS(LAG("CLOSE", 1) OVER (PARTITION BY Q."TICKER" ORDER BY Q."TIMESTAMP")) + 0.0000001) "B",
        "HIGH",
        "LOW",
        "CLOSE",
        "VOLUME",
        ROW_NUMBER() OVER (ORDER BY Q."TIMESTAMP") as rn
        FROM QUOTE Q
        WHERE Q."TICKER" = TICKER_NAME
        order by Q."TIMESTAMP"
    ),
    price_pivots AS (
        SELECT 
            curr.rn, 
            curr."TICKER", 
            curr."TIMESTAMP", 
            curr."CLOSE", 
            curr."HIGH", 
            curr."LOW",
            curr."VOLUME",            
	        curr."ΔP",
	        curr."ΔV",
	        curr."ΔH",
	        curr."ΔH2",
	        curr."ΔL",
	        curr."ΔL2",                
            (curr.rn - h_up.rn) AS i_h_up, 
            (h_up."HIGH" - curr."HIGH") AS p_h_up,
            (curr.rn - h_dn.rn) AS i_h_dn, 
            (curr."HIGH" - h_dn."HIGH") AS p_h_dn,
            (curr.rn - l_up.rn) AS i_l_up, 
            (l_up."LOW" - curr."LOW") AS p_l_up,
            (curr.rn - l_dn.rn) AS i_l_dn, 
            (curr."LOW" - l_dn."LOW") AS p_l_dn,
			("B" -  LAG("B", 2) OVER (ORDER BY curr."TIMESTAMP")) Bb, 
            (LEAD("B", 2) over (partition by curr."TICKER" order by curr."TIMESTAMP") -  "B") Bf,
			ABS(("B" - LAG("B", 1) OVER (ORDER BY curr."TIMESTAMP"))) + ABS(LAG("B", 1) OVER (ORDER BY curr."TIMESTAMP")- LAG("B", 2) OVER (ORDER BY curr."TIMESTAMP")) Fb, 
			ABS(LEAD("B", 2) over (partition by curr."TICKER" order by curr."TIMESTAMP") -  LEAD("B", 1) over (partition by curr."TICKER" order by curr."TIMESTAMP")) + ABS(LEAD("B", 1) over (partition by curr."TICKER" order by curr."TIMESTAMP") -  "B") Ff
			
        FROM BARS curr
        LEFT JOIN LATERAL (SELECT rn, "HIGH" FROM bars prev WHERE prev.rn < curr.rn AND prev."HIGH" > curr."HIGH" ORDER BY prev.rn DESC LIMIT 1) h_up ON TRUE
        LEFT JOIN LATERAL (SELECT rn, "HIGH" FROM bars prev WHERE prev.rn < curr.rn AND prev."HIGH" < curr."HIGH" ORDER BY prev.rn DESC LIMIT 1) h_dn ON TRUE
        LEFT JOIN LATERAL (SELECT rn, "LOW"  FROM bars prev WHERE prev.rn < curr.rn AND prev."LOW"  > curr."LOW"  ORDER BY prev.rn DESC LIMIT 1) l_up ON TRUE
        LEFT JOIN LATERAL (SELECT rn, "LOW"  FROM bars prev WHERE prev.rn < curr.rn AND prev."LOW"  < curr."LOW"  ORDER BY prev.rn DESC LIMIT 1) l_dn ON TRUE
    ),
    price_bases AS (
        SELECT *, GREATEST(i_h_up, i_h_dn, i_l_up, i_l_dn) AS B_t, GREATEST(p_h_up, p_h_dn, p_l_up, p_l_dn) AS C_t
        FROM price_pivots
    ),
    price_angles as (
     	SELECT 
     		rn,
	       price_bases."TICKER", 
	       price_bases."TIMESTAMP", 
	       price_bases."ΔP",
	       price_bases."ΔV",
	       price_bases."ΔH",
	       price_bases."ΔH2",
	       price_bases."ΔL",
	       price_bases."ΔL2",   
	       price_bases."VOLUME",
	       price_bases.Bb,
	       price_bases.Bf,
	       price_bases.Fb,
	       price_bases.Ff,
	       COALESCE(ATAN((1.0 * i_h_up / NULLIF(B_t, 0)) / ((1.0 * p_h_up / NULLIF(C_t, 0)) + 0.000009)), 0) AS "Θh↑",
	       COALESCE(ATAN((1.0 * i_h_dn / NULLIF(B_t, 0)) / ((1.0 * p_h_dn / NULLIF(C_t, 0)) + 0.000009)), 0) AS "Θh↓",
	       COALESCE(ATAN((1.0 * i_l_up / NULLIF(B_t, 0)) / ((1.0 * p_l_up / NULLIF(C_t, 0)) + 0.000009)), 0) AS "Θl↑",
	       COALESCE(ATAN((1.0 * i_l_dn / NULLIF(B_t, 0)) / ((1.0 * p_l_dn / NULLIF(C_t, 0)) + 0.000009)), 0) AS "Θl↓"
		FROM price_bases
    ),
    volume_pivots as (SELECT 
            curr."TICKER", curr."TIMESTAMP",
            curr."ΔP",
	        curr."ΔV",
	        curr."ΔH",
	        curr."ΔH2",
	        curr."ΔL",
	        curr."ΔL2", 
	        curr."Θh↑",
	        curr."Θh↓",
	        curr."Θl↑",
	        curr."Θl↓",	 
	        curr.Bb,
	        curr.Bf,	
	       	curr.Fb,
	       	curr.Ff,        
            (curr.rn - v_up.rn) AS i_v_up, 
            (v_up."VOLUME" - curr."VOLUME") AS val_v_up,
            (curr.rn - v_dn.rn) AS i_v_dn, 
            (curr."VOLUME" - v_dn."VOLUME") AS val_v_dn
        FROM price_angles curr
        LEFT JOIN LATERAL (
            SELECT rn, "VOLUME" FROM price_angles 
            WHERE rn < curr.rn AND "VOLUME" > curr."VOLUME" 
            ORDER BY rn DESC LIMIT 1
        ) v_up ON TRUE
        LEFT JOIN LATERAL (
            SELECT rn, "VOLUME" FROM price_angles 
            WHERE rn < curr.rn AND "VOLUME" < curr."VOLUME" 
            ORDER BY rn DESC LIMIT 1
        ) v_dn ON TRUE
     ),
     volume_bases as (
        SELECT 
            volume_pivots."TICKER",
            volume_pivots."TIMESTAMP",
            volume_pivots."ΔP",
	        volume_pivots."ΔV",
	        volume_pivots."ΔH",
	        volume_pivots."ΔH2",
	        volume_pivots."ΔL",
	        volume_pivots."ΔL2",  
	        volume_pivots.Bb,
	        volume_pivots.Bf,	
	       	volume_pivots.Fb,
	       	volume_pivots.Ff,     		        
	        volume_pivots."Θh↑",
	        volume_pivots."Θh↓",
	        volume_pivots."Θl↑",
	        volume_pivots."Θl↓",	           
            GREATEST(i_v_up, i_v_dn, 1) AS B_t,
            GREATEST(ABS(val_v_up), ABS(val_v_dn), 0.000009) AS C_t,
            i_v_up, 
            val_v_up,
            i_v_dn, 
            val_v_dn
        FROM volume_pivots  
     ) SELECT             
        	volume_bases."TICKER", 
        	volume_bases."TIMESTAMP",
            volume_bases."ΔP",
	        volume_bases."ΔV",
	        volume_bases."ΔH",
	        volume_bases."ΔH2",
	        volume_bases."ΔL",
	        volume_bases."ΔL2",	        
	        volume_bases."Θh↑",
	        volume_bases."Θh↓",
	        volume_bases."Θl↑",
	        volume_bases."Θl↓",	
            ATAN(COALESCE((1.0 * i_v_up / NULLIF(B_t, 0)) / ((1.0 * val_v_up / NULLIF(C_t, 0)) + 0.000009), 0)) AS "φ1",
            ATAN(COALESCE((1.0 * i_v_dn / NULLIF(B_t, 0)) / ((1.0 * val_v_dn / NULLIF(C_t, 0)) + 0.000009), 0)) AS "φ2",
 	        volume_bases.Bb,
	        volume_bases.Bf,	 
	       	volume_bases.Fb,
	       	volume_bases.Ff        
        FROM volume_bases
        ORDER BY volume_bases."TIMESTAMP";
end;
$$;

/* 
 	"TICKER" VARCHAR(20), 
	"TIMESTAMP" TIMESTAMP,
    "ΔP" DOUBLE PRECISION,
    "ΔV" DOUBLE PRECISION,
    "ΔH" DOUBLE PRECISION,
    "ΔH2" DOUBLE PRECISION,
    "ΔL" DOUBLE PRECISION,
    "ΔL2" DOUBLE PRECISION,	        
    "Θh↑" DOUBLE PRECISION,
    "Θh↓" DOUBLE PRECISION,
    "Θl↑" DOUBLE PRECISION,
    "Θl↓" DOUBLE PRECISION,	
    "φ1" DOUBLE PRECISION,
    "φ2" DOUBLE PRECISION,
    Bb DOUBLE PRECISION,
    Bf DOUBLE PRECISION	 
 * */