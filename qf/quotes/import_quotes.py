from datetime import datetime, timedelta
import traceback
import psycopg2
import yfinance as yf
import numpy as np


def table_name(quote_name):
    return "quote_stocks"

def quote_exists(connection, quote_name):
    cursor = connection.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name(quote_name)} WHERE ticker = %s", (quote_name,))
    records = cursor.fetchall()
    count = records[0][0]
    cursor.close()
    return count > 0

def import_yfinance_historical_data(connection, historical_data, quote_name):        
    cursor = connection.cursor();
    statement = f"PREPARE INSERT_QUOTE AS INSERT INTO {table_name(quote_name)} (ticker, quote_timestamp, open_price, high_price, low_price, close_price, volume) VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT (ticker, quote_timestamp) DO NOTHING;"
    cursor.execute(statement)
    count = 0

    for index, row in historical_data.iterrows():
        cursor.execute("EXECUTE INSERT_QUOTE (%s, %s, %s, %s, %s, %s, %s)", (quote_name, index, row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]))        
        connection.commit()
        count += 1
        if count % 100 == 0:
            print(f"👍: Inserted {count} rows for {quote_name}")

    cursor.execute("DEALLOCATE INSERT_QUOTE")
    cursor.close()
    return count

def import_last_10_years(connection, quote_name, interval):
    ticker_data = yf.Ticker(quote_name)
    print(interval)
    if interval == "1d":
        historical_data = ticker_data.history(period="10y")
    else:
        historical_data = ticker_data.history(period="3mo", interval=interval)

    import_yfinance_historical_data(connection, historical_data, quote_name)

def get_last_yfinance_quote(connection, quote_name):
    cursor = connection.cursor()
    cursor.execute(f"SELECT MAX(quote_timestamp) FROM {table_name(quote_name)} WHERE ticker = %s", (quote_name,))
    records = cursor.fetchall()
    last_quote = records[0][0]
    cursor.close()
    return last_quote

def import_since_last_update(connection, quote_name, interval):
    max_quote_date = get_last_yfinance_quote(connection, quote_name)
    first_quote_date = max_quote_date.date()
    today_date = datetime.now().date()
    tomorrows_date = today_date + timedelta(days=1)

    if today_date >= first_quote_date:
        ticker_data = yf.Ticker(quote_name)

        if interval != "1d":
            interval = "1h"
            first_quote_date = first_quote_date if tomorrows_date - first_quote_date <= timedelta(days=60) else first_quote_date - timedelta(days=60)

        historical_data = ticker_data.history(start=first_quote_date, end=tomorrows_date, interval=interval)        
        count = import_yfinance_historical_data(connection, historical_data, quote_name)
        print(f"📁: inserted {count} records from '{quote_name}' since '{first_quote_date}' to '{today_date}'.")
    else:
        print(f"📁: records for '{quote_name}' since '{today_date}' to '{first_quote_date}' are up to date.")

def import_yinance_quote(connection_string, quote, interval):
    connection = None

    try:
        connection = psycopg2.connect(connection_string)        
        if not quote_exists(connection, quote):
            import_last_10_years(connection, quote, interval)
        else:
            import_since_last_update(connection, quote, interval)

    except Exception as cause: 
        print(f"💥: Can't import {quote} using connection string f{connection_string}")
        traceback.print_exc()
    finally:
        if connection is not None:
            connection.close()


def import_yfinace_quotes(connection, connection_string, redis_connection, quotes, interval):
    redis_connection.flushdb()
    for quote_name in quotes:
        import_yinance_quote(connection_string, quote_name, interval)      
       