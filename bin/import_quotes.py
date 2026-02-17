
import json
import os
import sys
from qf.dbsync import update_structure, read_quote_names, db_config
from qf.quotes import import_quotes


if __name__ == "__main__":
    quotes_file = sys.argv[1]
    quotes = read_quote_names(quotes_file)
    lookback_periods = 14
    connection_string, _ = db_config()

    update_structure(connection_string)
    import_quotes(connection_string, quotes)
    

    