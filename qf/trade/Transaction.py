from typing import NamedTuple
import qf.trade
class Transaction(NamedTuple):
    ticker: str
    entry_index: int
    exit_index: int
    duration: int
    side: int
    entry_price: float
    exit_price: float
    pl: float # profit or loss
    take_profit: float
    stop_loss: float
    exit_reason: int # -1 stop loss, 0 bar close, 1 take profit
    friction: float
    entry_force: float
    t_l_up: float
    t_h_dn: float
    efficiency_ratio: float
    current_atr_pct: float
    phi_1: float
    phi_2: float
    W:float
    d:float
    
    @staticmethod
    def from_position(position, exit_index, current_open_price, current_low_price, current_high_price, current_close_price, current_atr_pct, scalping):
        if position is None:
            return None

        exit_reason = None
        exit_price = 0
        pl = 0
        
        # side 1 = Long
        if position.side == 1:
            if current_open_price >= position.take_profit or current_high_price >= position.take_profit:
                exit_price = position.take_profit 
                exit_reason = 1
                pl = position.take_profit - position.entry_price
            elif current_open_price <= position.stop_loss or current_low_price <= position.stop_loss:
                exit_price = min(position.stop_loss, current_open_price)
                exit_reason = -1
                pl = position.stop_loss - position.entry_price   
            elif scalping and exit_index > position.entry_index and current_close_price > position.entry_price:
                exit_price = current_close_price
                exit_reason = 0
                pl = current_close_price - position.entry_price
                
        # side -1 = Short
        elif position.side == -1:
            if current_open_price <= position.take_profit or current_low_price <= position.take_profit:
                exit_price = position.take_profit
                exit_reason = 1
                pl = position.entry_price - position.take_profit
            elif current_open_price >= position.stop_loss or current_high_price >= position.stop_loss:
                exit_price =  max(position.stop_loss, current_open_price) # position.stop_loss
                exit_reason = -1
                pl = position.entry_price - position.stop_loss
            elif scalping and exit_index > position.entry_index and current_close_price < position.entry_price:
                exit_price = current_close_price
                exit_reason = 0
                pl = position.entry_price - current_close_price

        if exit_reason is None:
            return None

        slip_in = qf.trade.dynamic_slippage(current_atr_pct)
        slip_out = qf.trade.dynamic_slippage(current_atr_pct)
        friction = (position.entry_price * slip_in) + (exit_price * slip_out)
        return Transaction(
            ticker = position.ticker,
            exit_index = exit_index,
            entry_index = position.entry_index,
            duration = exit_index - position.entry_index,
            side = position.side,
            entry_price = position.entry_price,
            exit_price = exit_price,
            pl = (pl-friction) * position.quantity,
            take_profit = position.take_profit, # Fixed name
            stop_loss = position.stop_loss, # Fixed name
            exit_reason = exit_reason,
            friction = friction,
            t_l_up = position.t_l_up,
            t_h_dn = position.t_h_dn,
            efficiency_ratio = position.efficiency_ratio,
            current_atr_pct = position.current_atr_pct,
            entry_force = position.entry_force,
            phi_1=position.phi_1,
            phi_2=position.phi_2,
            W = position.W,
            d = position.d        
        )