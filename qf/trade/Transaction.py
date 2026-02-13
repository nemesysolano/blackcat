from typing import NamedTuple
from .Position import Position

class Transaction(NamedTuple):
    ticker: str
    entry_index: int
    entry_price: float
    entry_force: float
    side: int # 1 or -1
    quantity: int
    take_profit: float
    stop_loss: float
    pl: float
    exit_index: int
    exit_price: float
    
    @staticmethod
    def from_position(position: Position, pl: float, exit_index: int, exit_price: float):
        return Transaction(
            ticker=position.ticker,
            entry_index=position.entry_index,
            entry_price=position.entry_price,
            entry_force=position.entry_force,
            side=position.side,
            quantity=position.quantity,
            take_profit=position.take_profit,
            stop_loss=position.stop_loss,
            pl=pl,
            exit_index=exit_index,
            exit_price=exit_price
        )

