 with direction as (
        select "TICKER", "TIMESTAMP", case when ABS("CLOSE" - "OPEN")  > 0.00000001 THEN  (("CLOSE" - "OPEN")/("HIGH" - "LOW")) else 0  end d from quote
        where "TICKER" = 'AAPL'
        order by "TIMESTAMP"
    )
    select "TICKER", "TIMESTAMP",
    (d - LAG(d, 1) OVER (order by "TIMESTAMP")) AS d_target
    from direction
    order by "TIMESTAMP"