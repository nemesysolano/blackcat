
from qf.indicators.wavelets import pricetime, volumetime
from qf.indicators import volumeprice
import numpy as np

def ensemble(sqlalchemy_url, quote_name, lookback_periods):
    pricetime_df, pricetime_features, pricetime_target = pricetime(sqlalchemy_url, quote_name, lookback_periods)
    volumetime_df, volumetime_features, volumetime_target = volumetime(sqlalchemy_url, quote_name, lookback_periods)
    volumeprice_df, volumeprice_features, volumeprice_target = volumeprice(sqlalchemy_url, quote_name, lookback_periods)

    merged_target = 'avg_target'
    merged_df = volumeprice_df

    merged_df[volumetime_features] = volumetime_df[volumetime_features]
    merged_df[volumetime_target] = volumetime_df[volumetime_target]

    merged_df[pricetime_features] = pricetime_df[pricetime_features]
    merged_df[pricetime_target] = pricetime_df[pricetime_target]

    merged_df.dropna(inplace=True)    
    merged_df[merged_target] = (50*merged_df[pricetime_target] + 30*merged_df[volumetime_target] + 20*merged_df[volumeprice_target])/100

    merged_features = []
    merged_features.extend(pricetime_features)
    merged_features.extend(volumetime_features)
    merged_features.extend(volumeprice_features)
    merged_df.dropna(inplace=True)
    return merged_df, merged_features, merged_target




    