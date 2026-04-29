import hashlib
from io import StringIO
import redis
import pandas as pd

def fetch_cached_csv(redis_connection, sql_template):
    hash_object = hashlib.sha256(sql_template.encode('utf-8'))
    hex_digest = hash_object.hexdigest()
    if redis_connection.exists(hex_digest):
        content = redis_connection.get(hex_digest)
        return pd.read_csv(StringIO(content), index_col='quote_timestamp', parse_dates=True)
    return None

def write_to_cache(redis_connection, sql_template, df):
    hash_object = hashlib.sha256(sql_template.encode('utf-8'))
    hex_digest = hash_object.hexdigest()
    csv_content = df.to_csv(index=True)
    redis_connection.set(hex_digest, csv_content)
    return hex_digest

def fetch(connection, redis_connection, sql_template):
    df = fetch_cached_csv(redis_connection, sql_template)
    if df is None:
        df = pd.read_sql(sql_template, connection)
        df.dropna(inplace=True)
        df.set_index('quote_timestamp', inplace=True)
        write_to_cache(redis_connection, sql_template, df)
    return df