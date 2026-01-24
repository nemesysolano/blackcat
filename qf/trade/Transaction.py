from typing import NamedTuple

class Transaction(NamedTuple):
    ticker: str
    entry_index: int
    exit_index: int
    duration: int
    side: int
    entry_price: float
    exit_price: float
    pl: float # profit or loss
    take_profit: float
    stop_loss: float
    exit_reason: int # -1 stop loss, 0 bar close, 1 take profit

    @staticmethod
    def from_position(position, exit_index, n, current_open_price, current_low_price, current_high_price, current_close_price):
        if position is None:
            return None

        exit_reason = None
        exit_price = 0
        pl = 0
        
        # side 1 = Long
        if position.side == 1:
            # Use position.take_profit instead of position.tp
            if current_high_price >= position.take_profit:
                exit_price = position.take_profit 
                exit_reason = 1
                pl = position.take_profit - position.entry_price
            elif current_low_price <= position.stop_loss or current_open_price <= position.stop_loss:
                exit_price = min(position.stop_loss, current_open_price)
                exit_reason = -1
                pl = position.stop_loss - position.entry_price                
        # side -1 = Short
        elif position.side == -1:
            if current_low_price <= position.take_profit:
                exit_price = position.take_profit
                exit_reason = 1
                pl = position.entry_price - position.take_profit
            elif current_high_price >= position.stop_loss or current_open_price >= position.stop_loss:
                exit_price =  max(position.stop_loss, current_open_price) # position.stop_loss
                exit_reason = -1
                pl = position.entry_price - position.stop_loss
        elif exit_index == n-1: # Means that we have an open position at end of stream
            if position.side == 1:
                exit_price = current_close_price
                exit_reason = 0
                pl = current_close_price - position.entry_price
            else:
                exit_price = current_close_price
                exit_reason = 0
                pl = position.entry_price - current_close_price

        if exit_reason is None:
            return None

        return Transaction(
            ticker = position.ticker,
            exit_index = exit_index,
            entry_index = position.entry_index,
            duration = exit_index - position.entry_index,
            side = position.side,
            entry_price = position.entry_price,
            exit_price = exit_price,
            pl = pl * position.quantity,
            take_profit = position.take_profit, # Fixed name
            stop_loss = position.stop_loss, # Fixed name
            exit_reason = exit_reason
        )