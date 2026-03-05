# FracDiffPosition.py
from datetime import datetime
import numpy as np
from typing import NamedTuple

class FracDiffPosition(NamedTuple):
    ticker: str
    entry_index: int
    entry_price: float
    Λ: float
    Λ_hat: float
    L: float
    L_hat: float
    side: int 
    quantity: int
    take_profit: float
    stop_loss: float
    state: list
    entry_date: datetime 