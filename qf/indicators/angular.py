from qf.indicators.augmentation import add_volatility_columns
from sqlalchemy import create_engine
import pandas as pd

def angular_sql(quote_name, lagged_target, target):
    features = {
            "ΔP",
	        "ΔV",
	        "ΔH",
	        "ΔH2",
	        "ΔL",
	        "ΔL2",	        
	        "Θh↑",
	        "Θh↓",
	        "Θl↑",
	        "Θl↓",	
            "φ1",
            "φ2",
 	        lagged_target	        
    }
    
    return (f"SELECT * FROM ANGULAR_INDICATORS('{quote_name}')", list(sorted(features)), target)

def angular_bar_sql(quote_name):
    return angular_sql(quote_name, "bb", "bf")

def angular_force_sql(quote_name):
    return angular_sql(quote_name, "fb", "ff")

def angular(sqlalchemy_url, quote_name, lagged_target, target):
    (sql_template, features, target) = angular_sql(quote_name, lagged_target, target)
    engine = create_engine(sqlalchemy_url)
    
    with engine.connect() as connection:
        df = pd.read_sql(sql_template, connection)
        # engine.connect() used in a 'with' block handles closing, 
        # but explicit close is fine.
        connection.close()
    engine.dispose()
    
    # Drop rows where LAG functions returned NULL (the beginning of the dataset)
    df.dropna(inplace=True)
    df.set_index('TIMESTAMP', inplace=True)
    # add_volatility_columns(df, lookback_periods)
    return df, features, target

def angular_bar(sqlalchemy_url, quote_name, lookback_periods):
    df, features, target = angular(sqlalchemy_url, quote_name, "bb", "bf")
    return df, features, target


def angular_force(sqlalchemy_url, quote_name, lookback_periods):
    df, features, target=  angular(sqlalchemy_url, quote_name, "fb", "ff")
    return df, features, target