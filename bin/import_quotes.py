
import redis
import sys

from sqlalchemy import create_engine
from qf.dbsync import update_structure, read_quote_names, db_config
from qf.quotes import import_yfinace_quotes


if __name__ == "__main__":
    quotes_file = sys.argv[1]
    interval = sys.argv[2] if len(sys.argv) > 2 else "1d"    
    quotes = read_quote_names(quotes_file)
    lookback_periods = 14
    connection_string, sqlalchemy_url, redis_host, redis_port, redis_database = db_config()
    redis_connection = redis.Redis(host=redis_host, port=redis_port, db=redis_database, decode_responses=True)
    engine = create_engine(sqlalchemy_url)
    with engine.connect() as connection:
        update_structure(connection_string)
        import_yfinace_quotes(connection, connection_string, redis_connection, quotes, interval)
        connection.close()
    engine.dispose()
    redis_connection.close()
    

    