import json
import sys
import os
from qf.dbsync import read_quote_names, db_config
import pandas as pd

def get_quotes(connection, quote_name, index):
    # Fetch OHLC + Volume to match X_test timestamps
    sql_template = f"""
        with quote_data as (
            SELECT "TICKER", "TIMESTAMP", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME",
            ("CLOSE" - LAG("CLOSE", 1) over (ORDER BY "TIMESTAMP")) / ("CLOSE" + LAG("CLOSE", 1) over (ORDER BY "TIMESTAMP")) "ΔP"  
            FROM "QUOTE" WHERE "TICKER" = '{quote_name}'
        ) select quote_data."TICKER", quote_data."TIMESTAMP", "OPEN", "HIGH", "LOW", "CLOSE", 
            COALESCE(STDDEV(quote_data."ΔP") OVER(ORDER BY quote_data."TIMESTAMP" ROWS BETWEEN 30 PRECEDING AND CURRENT ROW),0.0001) "δP" ,
            coalesce(POWER((AVG("VOLUME") OVER(ORDER BY quote_data."TIMESTAMP" ROWS BETWEEN 30 PRECEDING AND CURRENT ROW) - "VOLUME") / (AVG("VOLUME") OVER(ORDER BY quote_data."TIMESTAMP" ROWS BETWEEN 30 PRECEDING AND CURRENT ROW) + "VOLUME" + 0.000009),2),0) "V",
            AVG("VOLUME") OVER(ORDER BY quote_data."TIMESTAMP" ROWS BETWEEN 30 PRECEDING AND CURRENT ROW) "AV",
            indicators."H"
        from quote_data inner join ANGULAR_INDICATORS('{quote_name}') indicators on quote_data."TIMESTAMP" = indicators."TIMESTAMP"
        order by "TIMESTAMP" 
    """
    
    df = pd.read_sql(sql_template, connection)        
    df.set_index('TIMESTAMP', inplace=True)
    df = df.loc[df.index.intersection(index)] 
    df = df.sort_index()
    
    return df


def check_if_tradable(quote_stats):
    try:
        return quote_stats["Edge"] > 10 and quote_stats["tradable"]
    except:
        return False
    
def get_stats(model_stats, quote_name):
    try:
        return model_stats.loc[quote_name]
    except:
        return None

def get_mse(quote_stats):
    try:
         return quote_stats["MSE"]
    except:
        return None



if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print("Usage: python trade_wavelets.py <quotes_file> [predictor]")
            sys.exit(1)

        quotes_file = sys.argv[1]
        _, sqlalchemy_url = db_config()
        quotes = read_quote_names(quotes_file)
    except Exception as e:
        print(f"Error reading quotes: {e}")
        quotes = []

    # Load Reports.
    price_direction_stats = get_model_stats(os.getcwd(), "report-price-time-wavelet-direction.csv")
    volume_direction_stats = get_model_stats(os.getcwd(), "report-volume-time-wavelet-direction.csv")
    force_stats = get_model_stats(os.getcwd(), "report-price-time-wavelet-force.csv")
    lookback_periods = 14
    engine = create_engine(sqlalchemy_url)