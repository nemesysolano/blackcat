from datetime import datetime
import numpy as np
from typing import NamedTuple

class FracDiffTransaction(NamedTuple):
    entry_date: datetime
    ticker: str
    entry_index: int
    entry_price: float
    leverage: float
    Lambda: float
    Lambda_hat: float
    L: float
    L_hat: float
    side: int # 1 or -1
    quantity: int
    take_profit: float
    stop_loss: float
    signal: str
    pl: float
    cost: float
    net_pl: float
    exit_index: int
    exit_price: float
    exit_date: datetime
    exit_reason: int
    stallness_reason: str
    state: list    
    trailing: list

    @staticmethod
    def from_position(position, current_index, exit_price, exit_reason, profit_loss, exit_date, stallness_reason = None):
        # 1. Commission Per Side ($0.02 per share, $7 minimum)
        spread_rate = 0.02
        spread_commission = position.quantity * spread_rate # 0.02 USD per share
        platform_commission = 6 # cTrader commissions are charged as $6 USD round-trip.
        total_commission = spread_commission + platform_commission
        
        # 2. Trading Fees (Daily Swap Charge / Financing)
        # Formula: (market closing price x trade size x (our charge +/- ARR)) / 360
        pepperstone_charge = 0.025  # 2.5% fixed charge
        
        # Using a realistic current Alternative Reference Rate (ARR) like SOFR (approx 5.0% - 5.5%)
        arr = np.round(np.random.uniform(0.050, 0.056),3)
        
        # Directional Swap Logic:
        # Longs typically pay (Charge + ARR)
        # Shorts typically receive (ARR - Charge), which can be a credit (negative cost) or a small debit
        if position.side == 1:
            swap_rate = pepperstone_charge + arr
        else:
            swap_rate = pepperstone_charge - arr
            
        # Calculate holding duration
        holding_days = max(0, (exit_date - position.entry_date).days)
        
        swap_cost = 0.0
        if holding_days > 0:
            # Using the average trade price as a proxy for daily closing prices over the hold period
            avg_market_price = (position.entry_price + exit_price) / 2.0
            daily_swap = (avg_market_price * position.quantity * swap_rate) / 360.0
            swap_cost = daily_swap * holding_days

        # 3. Final Total Cost
        cost = total_commission + swap_cost
        net_pl = profit_loss - cost
        
        return FracDiffTransaction(
            entry_date = position.entry_date,
            ticker = position.ticker,
            entry_index = position.entry_index,
            entry_price = position.entry_price,
            leverage = position.leverage,
            Lambda = position.Lambda,
            Lambda_hat = position.Lambda_hat,
            L = position.L,
            L_hat = position.L_hat,
            side = position.side,
            quantity = position.quantity,
            take_profit = position.take_profit,
            stop_loss = position.stop_loss,
            signal = position.signal,
            pl = profit_loss,
            cost = cost,
            net_pl = net_pl,
            exit_index = current_index,
            exit_price = exit_price,
            exit_date = exit_date,
            exit_reason = exit_reason,
            stallness_reason = stallness_reason,
            state = position.state,
            trailing = position.trailing.copy()
        )