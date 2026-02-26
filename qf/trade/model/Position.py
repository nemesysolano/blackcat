# Position.py
from datetime import datetime
import numpy as np
from typing import NamedTuple

class Position(NamedTuple):
    ticker: str
    entry_index: int
    entry_price: float
    δf: float
    δP: float
    side: int 
    quantity: int
    take_profit: float
    stop_loss: float
    V: float
    H: float    
    state: list
    entry_date: datetime 