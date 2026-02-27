import pandas as pd
import numpy as np
from qf.dbsync import update_structure, read_quote_names, db_config
from qf.quotes.import_quotes import import_yfinace_quotes

if __name__ == "__main__":
    csv_url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    sp500_df = pd.read_csv(csv_url)
    quotes = sp500_df['Symbol'].tolist()
    connection_string, _ = db_config()

    update_structure(connection_string)
    import_yfinace_quotes(connection_string, quotes)