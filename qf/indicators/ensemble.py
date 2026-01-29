from qf.indicators.augmentation import add_volatility_columns
from qf.indicators.wavelets import pricetime, volumetime
from qf.indicators import volumeprice, bardirection
import numpy as np
import pandas as pd

def ensemble(sqlalchemy_url, quote_name, lookback_periods):
    # 1. Fetch all component indicators
    pt_df, pt_features, pt_target = pricetime(sqlalchemy_url, quote_name, lookback_periods)
    vt_df, vt_features, vt_target = volumetime(sqlalchemy_url, quote_name, lookback_periods)
    vp_df, vp_features, vp_target = volumeprice(sqlalchemy_url, quote_name, lookback_periods)
    bd_df, bd_features, bd_target = bardirection(sqlalchemy_url, quote_name, lookback_periods,)

    merged_target = 'avg_target'
    # Using pricetime as the base because it contains the structural 'Θ' angles
    merged_df = pt_df.copy()
    
    # 2. Merge Features and Targets
    # Use pd.concat to merge all features at once, aligning on index (inner join)
    # This ensures we only keep rows common to all indicators and avoids manual assignment
    additional_cols = [
        vt_df[vt_features + [vt_target,'v_mom', 'v(t)', 'φ1', 'φ2']],
        vp_df[vp_features + [vp_target]],
        bd_df[bd_features + [bd_target,'d']]
    ]
    
    merged_df = pd.concat([merged_df] + additional_cols, axis=1, join='inner')    
    merged_df['market_power'] = 32 * np.abs(merged_df['w']) * np.abs(merged_df['v_mom'])
    merged_df['avg_market_power'] = merged_df['market_power'].rolling(window=lookback_periods).mean()
    merged_df['std_market_power'] = merged_df['market_power'].rolling(window=lookback_periods).std()
    merged_df['rel_market_power'] = (merged_df['avg_market_power'] -  merged_df['market_power']) / merged_df['std_market_power']
    merged_df.dropna(inplace=True)    

    add_volatility_columns(merged_df, lookback_periods)
    
    # 4. New Weighted Target (Structural-First)
    # PriceTime (45%) + BarDir (30%) + VolumeTime (20%) + VolumePrice (5%)
    merged_df[merged_target] = (
        30 * merged_df[pt_target] +
        30 * merged_df[bd_target] +
        25 * merged_df[vt_target] +
        15 * merged_df[vp_target]
    ) / 100

    # 5. Full Feature Set for the DNN
    merged_features = ['rel_market_power', 'ER']
    merged_features.extend(pt_features)
    merged_features.extend(vt_features)
    merged_features.extend(vp_features)
    merged_features.extend(bd_features)    
    
    return merged_df, merged_features, merged_target