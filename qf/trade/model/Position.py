# Position.py
import numpy as np
from typing import NamedTuple

class Position(NamedTuple):
    ticker: str
    entry_index: int
    entry_price: float
    fval_delta: float
    entry_dp: float
    side: int # 1 or -1
    quantity: int
    take_profit: float
    stop_loss: float
    V: float
    H: float
    state: list
    is_runner: bool = False
