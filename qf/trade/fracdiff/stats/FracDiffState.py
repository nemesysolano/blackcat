# Position.py
import numpy as np
from typing import NamedTuple

class FracDiffState(NamedTuple):
    index: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    Lambda: float
    Lambda_hat: float
    L: float
    L_hat: float