
import json
import os
import sys
from qf.dbsync import update_structure, read_quote_names, db_config
from qf.quotes import import_yfinace_quotes
import pandas as pd
from sqlalchemy import create_engine


if __name__ == "__main__":
    quotes_file_name = sys.argv[1]
    quotes_file_name_split  = os.path.splitext(quotes_file_name)
    if len(quotes_file_name_split) > 1:
        export_file_name = os.path.basename(quotes_file_name).replace(quotes_file_name_split[1], ".csv")
    else:
        export_file_name = os.path.basename(quotes_file_name) + ".csv"
    
    export_file_name = os.path.join(os.getcwd(), "data", export_file_name)

    if os.path.exists(export_file_name):
        os.remove(export_file_name)

    quotes = read_quote_names(quotes_file_name)
    _, sqlalchemy_url = db_config()
    data_dir = os.path.join(os.getcwd(), "data")
    engine = create_engine(sqlalchemy_url)

    with open(export_file_name, 'w') as f:
        df = None
        with engine.connect() as connection:
            for quote in quotes:
                if df is None:
                    df = pd.read_sql(f"SELECT * FROM ANGULAR_INDICATORS('{quote}')", connection)
                else:
                    df = pd.concat([df, pd.read_sql(f"SELECT * FROM ANGULAR_INDICATORS('{quote}')", connection)], ignore_index=True)

                print(f"length of {quote} is {len(df)}")                        
            connection.close()
        df.to_csv(f, index=False)
        engine.dispose()

    