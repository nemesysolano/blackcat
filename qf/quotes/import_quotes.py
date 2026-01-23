from datetime import datetime, timedelta
import traceback
import psycopg2
import yfinance as yf

def quote_exists(connection, quote_name):
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM quote WHERE \"TICKER\" = %s", (quote_name,))
    records = cursor.fetchall()
    count = records[0][0]
    cursor.close()
    return count > 0

def import_historical_data(connection, historical_data, quote_name):
    cursor = connection.cursor();
    statement = "PREPARE INSERT_QUOTE AS INSERT INTO quote (\"TICKER\", \"TIMESTAMP\", \"OPEN\", \"HIGH\", \"LOW\", \"CLOSE\", \"VOLUME\") VALUES ($1, $2, $3, $4, $5, $6, $7)"
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
    import_historical_data(connection, historical_data, quote_name)

def get_last_quote(connection, quote_name):
    cursor = connection.cursor()
    cursor.execute("SELECT MAX(\"TIMESTAMP\") FROM quote WHERE \"TICKER\" = %s", (quote_name,))
    records = cursor.fetchall()
    last_quote = records[0][0]
    cursor.close()
    return last_quote

def import_since_last_update(connection, quote_name):
    last_quote = get_last_quote(connection, quote_name)
    next_quote = datetime.now()
    next_quote = next_quote.strftime('%Y-%m-%d')
    first_quote = last_quote.strftime('%Y-%m-%d')

    if next_quote < first_quote:
        ticker_data = yf.Ticker(quote_name)
        historical_data = ticker_data.history(start=next_quote, end=first_quote)
        count = import_historical_data(connection, historical_data, quote_name)
        print(f"📁: inserted {count} records from '{quote_name}' since '{first_quote}' to '{next_quote}.")
    else:
        print(f"📁: records for '{quote_name}' since '{first_quote}' to '{next_quote} are up to date.")

def import_quote(connection_string, quote):
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


def import_quotes(connection_string, quotes):
    for quote_name in quotes:
        import_quote(connection_string, quote_name)        