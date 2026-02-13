from sqlalchemy import create_engine
import pandas as pd
def read_quote_names(file_path):
    with open(file_path, 'r') as f:
        quote_names = [line.strip() for line in f]
    return quote_names

def read_angular_indicators(connection, sqlalchemy_url):
    engine = create_engine(sqlalchemy_url)
    
    with engine.connect() as connection:
        df = pd.read_sql("SELECT * FROM ALL_ANGULAR_INDICATORS", connection)
        # engine.connect() used in a 'with' block handles closing, 
        # but explicit close is fine.
        connection.close()
    engine.dispose()
    return df


