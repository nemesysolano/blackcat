with PCT_DIFF as (
            select      "TIMESTAMP", 
            "CLOSE", 
            ("CLOSE" - LAG("CLOSE", 1) OVER (order by "TIMESTAMP"))   / (ABS("CLOSE") + ABS( LAG("CLOSE",   1) OVER (order by "TIMESTAMP")) + 0.000009) "CLOSE_DIFF%%",
            "VOLUME",
            ("VOLUME" - LAG("VOLUME", 1) OVER (order by "TIMESTAMP")) / (ABS("VOLUME") + ABS( LAG("VOLUME", 1) OVER (order by "TIMESTAMP")) + 0.000009) "VOLUME_DIFF%%"
            FROM QUOTE
            where "TICKER" = 'AAPL'
            order by "TIMESTAMP"
        ),
        PRICE_VOL as (
            select "TIMESTAMP",
                "CLOSE_DIFF%%" * ABS("VOLUME_DIFF%%") y
            from PCT_DIFF
            order by "TIMESTAMP"
        )
        select "TIMESTAMP", 
        y - LAG(Y, 1) OVER (order by "TIMESTAMP") AS y_target,
        LAG(y, 1) OVER (order by "TIMESTAMP") - LAG(Y, 2) OVER (order by "TIMESTAMP") y1,LAG(y, 2) OVER (order by "TIMESTAMP") - LAG(Y, 3) OVER (order by "TIMESTAMP") y2,LAG(y, 3) OVER (order by "TIMESTAMP") - LAG(Y, 4) OVER (order by "TIMESTAMP") y3,LAG(y, 4) OVER (order by "TIMESTAMP") - LAG(Y, 5) OVER (order by "TIMESTAMP") y4,LAG(y, 5) OVER (order by "TIMESTAMP") - LAG(Y, 6) OVER (order by "TIMESTAMP") y5,LAG(y, 6) OVER (order by "TIMESTAMP") - LAG(Y, 7) OVER (order by "TIMESTAMP") y6,LAG(y, 7) OVER (order by "TIMESTAMP") - LAG(Y, 8) OVER (order by "TIMESTAMP") y7,LAG(y, 8) OVER (order by "TIMESTAMP") - LAG(Y, 9) OVER (order by "TIMESTAMP") y8,LAG(y, 9) OVER (order by "TIMESTAMP") - LAG(Y, 10) OVER (order by "TIMESTAMP") y9,LAG(y, 10) OVER (order by "TIMESTAMP") - LAG(Y, 11) OVER (order by "TIMESTAMP") y10,LAG(y, 11) OVER (order by "TIMESTAMP") - LAG(Y, 12) OVER (order by "TIMESTAMP") y11,LAG(y, 12) OVER (order by "TIMESTAMP") - LAG(Y, 13) OVER (order by "TIMESTAMP") y12,LAG(y, 13) OVER (order by "TIMESTAMP") - LAG(Y, 14) OVER (order by "TIMESTAMP") y13,LAG(y, 14) OVER (order by "TIMESTAMP") - LAG(Y, 15) OVER (order by "TIMESTAMP") y14
        from PRICE_VOL 
        order by "TIMESTAMP