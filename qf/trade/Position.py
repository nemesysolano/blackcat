# Position.py
import numpy as np
from typing import NamedTuple

class Position(NamedTuple):
    ticker: str
    entry_index: int
    entry_price: float
    entry_force: float
    side: int # 1 or -1
    quantity: int
    take_profit: float
    stop_loss: float
    state: list
    is_runner: bool = False