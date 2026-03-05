from datetime import datetime
import numpy as np
from typing import NamedTuple

class FracDiffTransaction(NamedTuple):
    entry_date: datetime
    ticker: str
    entry_index: int
    entry_price: float
    Λ: float
    Λ_hat: float
    L: float
    L_hat: float
    side: int # 1 or -1
    quantity: int
    take_profit: float
    stop_loss: float
    pl: float
    exit_index: int
    exit_price: float
    exit_date: datetime
    exit_reason: int
    stallness_reason: str
    state: list    

    @staticmethod
    def from_position(position, current_index, exit_price, exit_reason, profit_loss, exit_date, stallness_reason = None):
        return FracDiffTransaction(
            entry_date = position.entry_date,
            ticker = position.ticker,
            entry_index = position.entry_index,
            entry_price = position.entry_price,
            Λ = position.Λ,
            Λ_hat = position.Λ_hat,
            L = position.L,
            L_hat = position.L_hat,
            side = position.side,
            quantity = position.quantity,
            take_profit = position.take_profit,
            stop_loss = position.stop_loss,
            pl = profit_loss,
            exit_index = current_index,
            exit_price = exit_price,
            exit_date = exit_date,
            exit_reason = exit_reason,
            stallness_reason = stallness_reason,
            state = position.state
        )