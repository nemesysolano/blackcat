from datetime import datetime
from typing import NamedTuple
from .Position import Position

class Transaction(NamedTuple):
    entry_date: datetime
    ticker: str
    entry_index: int
    entry_price: float
    δP: float
    δf: float
    side: int # 1 or -1
    quantity: int
    take_profit: float
    stop_loss: float
    pl: float
    exit_index: int
    exit_price: float
    exit_reason: str    
    V: float
    H: float
    exit_date: datetime
    state: list
    @staticmethod
    def from_position(position: Position, pl: float, exit_index: int, exit_price: float, exit_reason: int, t: datetime):
        return Transaction(          
            entry_date = position.entry_date,          
            ticker=position.ticker,
            entry_index=position.entry_index,
            entry_price=position.entry_price,
            δP = position.δP,
            δf = position.δf,            
            side=position.side,
            quantity=position.quantity,
            take_profit=position.take_profit,
            stop_loss=position.stop_loss,
            pl=pl,
            exit_index=exit_index,
            exit_price=exit_price,
            exit_reason=exit_reason,
            V=position.V,
            H=position.H,
            exit_date=t,
            state = position.state
        )

