from qf.indicators.wavelets import pricetime, volumetime
from qf.indicators import volumeprice, bardirection
import numpy as np

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
    merged_df[vt_features] = vt_df[vt_features]
    merged_df[vt_target] = vt_df[vt_target]

    merged_df[vp_features] = vp_df[vp_features]
    merged_df[vp_target] = vp_df[vp_target]

    merged_df[bd_features] = bd_df[bd_features]
    merged_df[bd_target] = bd_df[bd_target]

    merged_df['d'] = bd_df['d']
    merged_df['v_mom'] = vt_df['v_mom']
    merged_df['market_power'] = 32 * np.abs(merged_df['w']) * np.abs(merged_df['v_mom']) 

    merged_df.dropna(inplace=True)    

    direction = abs(bd_df['CLOSE'] - bd_df['CLOSE'].shift(lookback_periods))
    volatility = abs(bd_df['CLOSE'] - bd_df['CLOSE'].shift(1)).rolling(window=lookback_periods).sum()
    merged_df['ER'] = direction / (volatility + 1e-9)
    merged_df.dropna(inplace=True)

    # 4. New Weighted Target (Structural-First)
    # PriceTime (45%) + BarDir (30%) + VolumeTime (20%) + VolumePrice (5%)
    merged_df[merged_target] = (
        45 * merged_df[pt_target] + 
        30 * merged_df[bd_target] + 
        20 * merged_df[vt_target] + 
        5  * merged_df[vp_target]
    ) / 100

    # 5. Full Feature Set for the DNN
    merged_features = ['market_power']
    merged_features.extend(pt_features)
    merged_features.extend(vt_features)
    merged_features.extend(vp_features)
    merged_features.extend(bd_features)
    
    return merged_df, merged_features, merged_target