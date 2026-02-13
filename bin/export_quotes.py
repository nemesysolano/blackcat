
import json
import os
import sys
from qf.dbsync import update_structure, read_quote_names, db_config
from qf.quotes import import_quotes
import pandas as pd
from sqlalchemy import create_engine


if __name__ == "__main__":
    quotes_file = sys.argv[1]
    quotes = read_quote_names(quotes_file)
    lookback_periods = 14
    _, sqlalchemy_url = db_config()
    data_dir = os.path.join(os.getcwd(), "data")
    engine = create_engine(sqlalchemy_url)
    
    with engine.connect() as connection:
        for quote in quotes:
            quote_file = os.path.join(data_dir, f"{quote}.csv")
            if os.path.exists(quote_file):
                print(f"{quote} already exists")
                continue
            df = pd.read_sql(f"SELECT * FROM ANGULAR_INDICATORS('{quote}')", connection)
            print(f"length of {quote} is {len(df)}")                        
            df.dropna(inplace=True)
            df.to_csv(quote_file, index=False)
        connection.close()
    engine.dispose()

    