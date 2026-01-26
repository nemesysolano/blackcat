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
    def create_with_check(quote_name, context, entry_index, prediction, edge, entry_price, contrarian):
        t = context.index[entry_index]
        
        # 1. Base Direction from Ensemble Model
        side = int(np.sign(prediction)) * (-1 if contrarian else 1)
        
        current_bar_dir = context.loc[t, "d"]
        # VETO: If the model predicts Long (+1), but the bar is closing as a 
        # Red/Down bar (negative 'd'), the trade is cancelled.
        if np.sign(current_bar_dir) != side:
            return None

        # 2. Extract Structural Angles & Bar Direction
        t_l_up = context.loc[t, "Θl↑"]
        t_h_dn = context.loc[t, "Θh↓"]
        # Assuming 'd' is the column name for Bar Direction (Close-Open)/(High-Low)
        current_bar_dir = context.loc[t, "d"] 

        # 3. Structural Sanity Check (Pricetime logic)
        volume_factor = 0.0
        if side == 1: # Long
            if t_l_up > t_h_dn: volume_factor = 1.0
            elif t_l_up > 0: volume_factor = 0.5 # Weak momentum
        else: # Short
            if t_h_dn > t_l_up: volume_factor = 1.0
            elif t_h_dn > 0: volume_factor = 0.5

        # 4. Bar Direction Conviction Filter (The New Step)
        # If bar direction is opposite to our prediction, it's a "fake out"
        if np.sign(current_bar_dir) != side:
            return None # Skip: Price is moving against the trend within the bar
        
        # Scale volume further if the bar is "weak" (less than 30% conviction)
        if abs(current_bar_dir) < 0.3:
            volume_factor *= 0.5

        # 5. VETO: If power is too low, the move is likely a 'fake' or 'drift'
        # A threshold of 0.15 - 0.20 is a good starting point for 'Laminar Flow'
        if "market_power" in context.columns:
            market_power = context.loc[t, "market_power"]
            if market_power < 0.0075:
                return None

        # 6. Final Sizing & Risk
        k_max = edge / 50
        
        if "ER" in context.columns:
            er_value = context.loc[t, "ER"]
            volatility_scaler = np.clip(er_value, 0.5, 1.0)        
        else:
            volatility_scaler = 1.0

        quantity = max(1, int(k_max * volume_factor * volatility_scaler * 100))
        take_profit = entry_price * (1 + 0.02 * side)
        stop_loss = entry_price * (1 - 0.01 * side)

        return Position(quote_name, entry_index, entry_price, side, quantity, take_profit, stop_loss)
    
    @staticmethod
    def create_without_check(quote_name, context, entry_index, prediction, edge, entry_price, contrarian):
        """
        Pure signal execution. No structural filters, no bar direction VETO.
        Used to establish a baseline for the model's raw predictive power.
        """
        # 1. Base Direction from Ensemble Model (allowing for contrarian flipping)
        side = int(np.sign(prediction)) * (-1 if contrarian else 1)
        
        # 2. Fixed Volume Factor
        # Without checks, we assume 100% size conviction based on the model signal
        volume_factor = 1.0

        # 3. Position Sizing
        # Formula: K_max = edge / 50
        k_max = edge / 50
        quantity = max(1, int(k_max * volume_factor * 100)) 

        # 4. Risk Management (Fixed 2% TP / 1% SL)
        take_profit = entry_price * (1 + 0.02 * side)
        stop_loss = entry_price * (1 - 0.01 * side)

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
