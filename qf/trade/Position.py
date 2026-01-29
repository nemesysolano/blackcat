import numpy as np
from typing import NamedTuple
RADIAN_THRESHOLD = np.radians(5.0)
FORCE_THRESHOLD_LOW = 0.70
FORCE_THRESHOLD_MIDDLE = 0.75
FORCE_THRESHOLD_HIGH = 0.85

class Position(NamedTuple):
    ticker: str
    entry_index: int
    entry_price: float
    entry_force: float
    side: int # 1 or -1
    quantity: int
    take_profit: float
    stop_loss: float
    t_l_up: float
    t_h_dn: float
    efficiency_ratio: float
    current_atr_pct: float    
    phi_1: float
    phi_2: float
    W: float
    d: float
    @staticmethod
    def _calculate_dynamic_levels(entry_price, side, edge, atr_pct, efficiency_ratio):
        """
        Calculates TP/SL based on market rhythm and model confidence.
        """
        reward = 0.02
        loss = 0.01

        if side == 1:
            tp_factor = 1 + reward
            sl_factor = 1 - loss
        else:
            tp_factor = 1 - reward
            sl_factor = 1 + loss

        take_profit = entry_price * tp_factor
        stop_loss = entry_price * sl_factor

        return take_profit, stop_loss

    @staticmethod
    def create_with_check(quote_name, context, entry_index, prediction, edge, entry_price, contrarian):
        t = context.index[entry_index]       
        side = int(np.sign(prediction)) * (-1 if contrarian else 1)
        efficiency_ratio = context.loc[t, "ER"]
        current_atr_pct = context.loc[t, "ATR %"]
    
        # 1. Calculate Average Wavelet Force (Structural Resonance)
        # Note: pt_df uses 'W' and vt_df uses 'v_mom' as the score columns
        
        w_t = context.loc[t, "w(t)"]
        v_t = context.loc[t, "v(t)"] 
        a_t = (w_t + v_t) / 2

        # VETO: Require at least 50% structural resonance to overcome friction
        # This replaces the need for an arbitrary prediction magnitude filter        
        if a_t < FORCE_THRESHOLD_LOW or FORCE_THRESHOLD_MIDDLE <= a_t < FORCE_THRESHOLD_HIGH:
            return None
        
        # 2. Extract Structural and Volume Angles
        t_l_up = context.loc[t, "Θl↑"]  # Support slope
        t_h_dn = context.loc[t, "Θh↓"]  # Resistance slope
        phi_1 = context.log[t, "φ1"]
        phi_2 = context.log[t, "φ2"]
        W = context.loc[t, "W"]
        d = context.loc[t, "d"]
        # --- PRICE STRUCTURE VETO ---
        if side == 1 and t_h_dn < -RADIAN_THRESHOLD:
            return None  # Veto Long: Resistance is dropping too sharply
        if side == -1 and t_l_up > RADIAN_THRESHOLD:
            return None  # Veto Short: Support is rising too sharply
        
        # 2. Adaptive Volume Factor
        # Scale size down if volatility is spiked (Turbulence protection)
        # If ATR is 2x the normal (0.005), we halve the position size
        vol_buffer = np.clip(0.005 / current_atr_pct, 0.2, 1.0)
        
        # 3. Position Sizing
        k_max = (edge / 50) * efficiency_ratio * vol_buffer
        quantity = max(1, int(k_max * 100)) 

        # 4. Risk Management (Fixed 2% TP / 1% SL)
        take_profit, stop_loss = Position._calculate_dynamic_levels(
            entry_price, side, edge, current_atr_pct, efficiency_ratio
        )

        return Position(
            ticker=quote_name,
            entry_index=entry_index,
            entry_price=entry_price,
            side=side,
            quantity=quantity,
            take_profit=take_profit,
            stop_loss=stop_loss,
            entry_force=a_t,
            t_l_up=t_l_up,
            t_h_dn=t_h_dn,
            efficiency_ratio=efficiency_ratio,
            current_atr_pct=current_atr_pct,
            phi_1 = phi_1,
            phi_2 = phi_2,
            W = W,
            d = d            
        )
    
    @staticmethod
    def create_without_check(quote_name, context, entry_index, prediction, edge, entry_price, contrarian):
        """
        Pure signal execution. No structural filters, no bar direction VETO.
        Used to establish a baseline for the model's raw predictive power.
        """
        # 1. Base Direction from Ensemble Model (allowing for contrarian flipping)
        t = context.index[entry_index]
        side = int(np.sign(prediction)) * (-1 if contrarian else 1)
        efficiency_ratio = context.loc[t, "ER"]
        current_atr_pct = context.loc[t, "ATR %"]

        # 2. Fixed Volume Factor
        # Without checks, we assume 100% size conviction based on the model signal
        volume_factor = 1.0

        # 3. Position Sizing
        # Formula: K_max = edge / 50
        k_max = (edge / 50) * efficiency_ratio
        quantity = max(1, int(k_max * volume_factor * 100)) 

        # 4. Risk Management (Fixed 2% TP / 1% SL)
        take_profit, stop_loss = Position._calculate_dynamic_levels(
            entry_price, side, edge, current_atr_pct, efficiency_ratio
        )

        w_t = context.loc[t, "w(t)"]
        v_t = context.loc[t, "v(t)"] 
        a_t = (w_t + v_t) / 2
        t_l_up = context.loc[t, "Θl↑"]  # Support slope
        t_h_dn = context.loc[t, "Θh↓"]  # Resistance slope
        phi_1 = context.loc[t, "φ1"]
        phi_2 = context.loc[t, "φ2"]
        W = context.loc[t, "w"]
        d = context.loc[t, "d"]

        return Position(
            ticker=quote_name,
            entry_index=entry_index,
            entry_price=entry_price,
            side=side,
            quantity=quantity,
            take_profit=take_profit,
            stop_loss=stop_loss,
            entry_force=a_t,
            t_l_up=t_l_up,
            t_h_dn=t_h_dn,
            efficiency_ratio=efficiency_ratio,
            current_atr_pct=current_atr_pct   ,
            phi_1 = phi_1,
            phi_2 = phi_2,
            W = W,
            d = d
        )

    @staticmethod
    def create(quote_name, context, entry_index, prediction, edge, entry_price, contrarian,  structural_check = True):
        if structural_check:
            return Position.create_with_check(quote_name, context, entry_index, prediction, edge, entry_price, contrarian)
        else:
            return Position.create_without_check(quote_name, context, entry_index, prediction, edge, entry_price, contrarian)
