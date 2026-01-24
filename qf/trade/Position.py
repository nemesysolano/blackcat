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
    def create(quote_name, context, entry_index, prediction, edge, entry_price):
        """
        Creates a new Position object.
        Currently focuses on plain entries, ignoring price-time angle sanity checks.
        """
        # t is the timestamp index for the current row
        t = context.index[entry_index]
        
        # 1. Base Direction from Momentum Model
        # The sign of the prediction tells us the side (Long = 1, Short = -1)
        side = int(np.sign(prediction))
        
        # 2. Sanity Check / Volume Factor
        # Angles check is disabled for now; defaulting to full size (1.0)
        volume_factor = 1.0

        # 3. Position Sizing
        # Formula: K_max = edge / 50
        k_max = edge / 50
        quantity = max(1, int(k_max * volume_factor * 100)) 

        # 4. Risk Management (TP/SL)
        # Using a fixed 2% profit target and 1% stop loss from entry
        take_profit = entry_price * (1 + 0.02 * side)
        stop_loss = entry_price * (1 - 0.01 * side)

        return Position(
            ticker = quote_name,
            entry_index = entry_index,
            entry_price = entry_price,
            side = side,
            quantity = quantity,
            take_profit = take_profit,
            stop_loss = stop_loss
        )