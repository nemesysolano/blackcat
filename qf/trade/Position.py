import numpy as np
from typing import NamedTuple

class Position(NamedTuple):
    ticker: str
    entry_index: int
    entry_price: float
    side: int # 1 or -1
    quantity: int
    take_profit: float
    stop_loss: float

    @staticmethod
    def _calculate_dynamic_levels(entry_price, side, edge, atr_pct, efficiency_ratio):
        """
        Calculates TP/SL based on market rhythm and model confidence.
        """
        reward = min(efficiency_ratio + (edge / 100), 0.975)
        print(f"Reward: {reward}")
        if side == 1:
            tp_factor = 1 + reward
            sl_factor = 1 - reward*0.25
        else:
            tp_factor = 1 - reward
            sl_factor = 1 + reward*0.25
        

        take_profit = entry_price * tp_factor
        stop_loss = entry_price * sl_factor

        
        return take_profit, stop_loss

    @staticmethod
    def create_with_check(quote_name, context, entry_index, prediction, edge, entry_price, contrarian):
        t = context.index[entry_index]
        side = int(np.sign(prediction)) * (-1 if contrarian else 1)
        efficiency_ratio = context.loc[t, "ER"]
        current_atr_pct = context.loc[t, "ATR %"]

        # 1. Structural VETO (The "Must-Have" conditions)
        t_l_up = context.loc[t, "Θl↑"]
        t_h_dn = context.loc[t, "Θh↓"]
        current_bar_dir = context.loc[t, "d"]
        
        # Basic structural check: Are we swimming with the current?
        structural_alignment = (side == 1 and t_l_up > t_h_dn) or (side == -1 and t_h_dn > t_l_up)
        bar_alignment = np.sign(current_bar_dir) == side

        if not (structural_alignment and bar_alignment):
            return None # Cancel trade if structure or bar direction opposes us

        # 2. Dynamic Market Power Factor (The "Conviction" scaling)
        market_power = context.loc[t, "market_power"]
        avg_power = context.loc[t, "avg_market_power"]
        
        # Ratio of current power to average (clamped between 0.5 and 1.0)
        power_ratio = market_power / (avg_power + 1e-9)
        volume_factor = np.clip(power_ratio, 0.5, 1.0)

        # 3. Position Sizing
        k_max = (edge / 50) * efficiency_ratio
        quantity = max(1, int(k_max * volume_factor * 100))

        # 4. Risk Management (Fixed 2% TP / 1% SL)
        take_profit, stop_loss = Position._calculate_dynamic_levels(
            entry_price, side, edge, current_atr_pct, efficiency_ratio
        )

        return Position(quote_name, entry_index, entry_price, side, quantity, take_profit, stop_loss)
    
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

        return Position(
            ticker=quote_name,
            entry_index=entry_index,
            entry_price=entry_price,
            side=side,
            quantity=quantity,
            take_profit=take_profit,
            stop_loss=stop_loss
        )

    @staticmethod
    def create(quote_name, context, entry_index, prediction, edge, entry_price, contrarian,  structural_check = True):
        if structural_check:
            return Position.create_with_check(quote_name, context, entry_index, prediction, edge, entry_price, contrarian)
        else:
            return Position.create_without_check(quote_name, context, entry_index, prediction, edge, entry_price, contrarian)
