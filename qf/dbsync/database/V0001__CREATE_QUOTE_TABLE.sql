CREATE TABLE IF NOT EXISTS  quote_stocks (
    quote_timestamp TIMESTAMP NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    open_price DECIMAL(10,2) NOT NULL,
    high_price DECIMAL(10,2) NOT NULL,
    low_price DECIMAL(10,2) NOT NULL,
    close_price DECIMAL(10,2) NOT NULL,
    volume BIGINT NOT NULL,
    CONSTRAINT QUOTE_PK_STOCKS PRIMARY KEY (quote_timestamp, ticker)
);
