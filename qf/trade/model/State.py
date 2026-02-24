# Position.py
import numpy as np
from typing import NamedTuple

class State(NamedTuple):
    index: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    δP: float
    V: float
    H: float
    fval_delta: float
    