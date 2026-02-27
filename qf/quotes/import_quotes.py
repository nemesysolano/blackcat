from datetime import datetime, timedelta
import traceback
import psycopg2
import yfinance as yf
import numpy as np

def table_name(quote_name):
    return "QUOTE_FOREX" if quote_name.endswith("=X") else "QUOTE_STOCKS"

def quote_exists(connection, quote_name):
    cursor = connection.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM \"{table_name(quote_name)}\" WHERE \"TICKER\" = %s", (quote_name,))
    records = cursor.fetchall()
    count = records[0][0]
    cursor.close()
    return count > 0

def import_yfinance_historical_data(connection, historical_data, quote_name, ticker):        
    cursor = connection.cursor();
    statement = f"PREPARE INSERT_QUOTE AS INSERT INTO \"{table_name(quote_name)}\" (\"TICKER\", \"TIMESTAMP\", \"OPEN\", \"HIGH\", \"LOW\", \"CLOSE\", \"VOLUME\") VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT (\"TICKER\", \"TIMESTAMP\") DO NOTHING;"
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

def import_last_10_years(connection, quote_name):
    ticker_data = yf.Ticker(quote_name)
    historical_data = ticker_data.history(period="10y")
    import_yfinance_historical_data(connection, historical_data, quote_name, ticker_data)

def get_last_yfinance_quote(connection, quote_name):
    cursor = connection.cursor()
    cursor.execute(f"SELECT MAX(\"TIMESTAMP\") FROM \"{table_name(quote_name)}\" WHERE \"TICKER\" = %s", (quote_name,))
    records = cursor.fetchall()
    last_quote = records[0][0]
    cursor.close()
    return last_quote

def import_since_last_update(connection, quote_name):
    max_quote_date = get_last_yfinance_quote(connection, quote_name)
    first_quote_date = max_quote_date + timedelta(days=1)
    next_quote_date = datetime.now()
    
    if next_quote_date > first_quote_date:
        ticker_data = yf.Ticker(quote_name)
        historical_data = ticker_data.history(start=first_quote_date, end=next_quote_date)
        count = import_yfinance_historical_data(connection, historical_data, quote_name, ticker_data)
        print(f"📁: inserted {count} records from '{quote_name}' since '{first_quote_date}' to '{next_quote_date}.")
    else:
        print(f"📁: records for '{quote_name}' since '{first_quote_date}' to '{next_quote_date} are up to date.")

def import_yinance_quote(connection_string, quote):
    connection = None

    try:
        connection = psycopg2.connect(connection_string)        
        if not quote_exists(connection, quote):
            import_last_10_years(connection, quote)
        else:
            import_since_last_update(connection, quote)

    except Exception as cause: 
        print(f"💥: Can't import {quote} using connection string \"f{connection_string}\"")
        traceback.print_exc()
    finally:
        if connection is not None:
            connection.close()


def import_yfinace_quotes(connection_string, quotes):
    for quote_name in quotes:
        import_yinance_quote(connection_string, quote_name)        