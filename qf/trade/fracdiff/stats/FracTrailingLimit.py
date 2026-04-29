# Position.py
import numpy as np
from typing import NamedTuple
from datetime import datetime

class FracTrailingLimit(NamedTuple):
    datetime: datetime
    take_profit: float
    stop_loss: float