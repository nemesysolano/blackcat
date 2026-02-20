
import json
import os
import numpy as np
exit_reasons = { '-1': 'Stop Loss', '0': 'Early Stop', '1': 'Take Profit'}
   
def get_returns(equity_array):
    if len(equity_array) > 1:
        return np.diff(equity_array) / equity_array[:-1]
    else:
        return np.array([0])
    
def create_backtest_stats(quote_name, equity_curve, long_trades, short_trades, winner_longs, winner_shorts, loser_longs, loser_shorts, transactions):
    equity_array = np.ravel(equity_curve)
    prev_equity = equity_array[:-1]    

    # Initialize returns with 0
    returns = np.zeros_like(prev_equity)
    
    # Only calculate for indices where capital was still above zero
    valid_mask = prev_equity > 0
    returns[valid_mask] = np.diff(equity_array)[valid_mask] / prev_equity[valid_mask]
    
    volatility = np.std(returns) if len(returns) > 0 else 0
    # If final_capital was an array due to the previous bug, take the first value
    f_cap = equity_array[-1]
    
    initial_capital = equity_array[0]
    total_return_pct = (f_cap - initial_capital) / initial_capital
    
    # Now np.diff will produce a simple 1D vector
    returns = get_returns(equity_array)
    
    # Standard deviation of returns (volatility)
    volatility = np.std(returns) if len(returns) > 0 else 0
    
    # Sharpe Ratio (Assuming 0 risk-free rate for simplicity)
    # Annualization factor depends on the timeframe of your data (e.g., np.sqrt(252))
    sharpe_ratio = (np.mean(returns) / volatility) if volatility != 0 else 0

    # 3. Drawdown Analysis
    # Peak equity reached up to each point in time
    running_max = np.maximum.accumulate(equity_array)
    drawdowns = (equity_array - running_max) / running_max
    max_drawdown = np.min(drawdowns)

    # 4. Summary Statistics
    stats = {
        "Ticker": quote_name,
        "Initial Capital": initial_capital,
        "Final Capital": f_cap,
        "Total Return (%)": total_return_pct * 100,
        "Max Drawdown (%)": max_drawdown * 100,
        "Volatility (per step)": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Number of Steps": len(equity_array),
        "Peak Equity": np.max(equity_array),
        "Final Drawdown (%)": drawdowns[-1] * 100,
        "Long Trades": long_trades,
        "Short Trades": short_trades,
        "Winner Longs": winner_longs,
        "Winner Shorts": winner_shorts,
        "Loser Longs": loser_longs,
        "Loser Shorts": loser_shorts
    }
    return stats, transactions

def write_results(output_file, details_file, stats, transactions):
    mode = 'a' if os.path.exists(output_file) else 'w'
    with open(output_file, mode) as f:
        if mode == 'w':
            print(
                "Ticker, Initial Capital, Final Capital, Total Return (%), Max Drawdown (%), Volatility (per step), Sharpe Ratio, Number of Steps, Peak Equity, Final Drawdown (%), Long Trades, Short Trades, Winner Longs, Winner Shorts, Loser Longs, Loser Shorts",
                file=f
            )

        print(
            f"{stats['Ticker']}, {stats['Initial Capital']:.2f}, {stats['Final Capital']:.2f}, {stats['Total Return (%)']:.2f}, {stats['Max Drawdown (%)']:.2f}, {stats['Volatility (per step)']:.2f}, {stats['Sharpe Ratio']:.2f}, {stats['Number of Steps']}, {stats['Peak Equity']:.2f}, {stats['Final Drawdown (%)']:.2f}, {stats['Long Trades']}, {stats['Short Trades']}, {stats['Winner Longs']}, {stats['Winner Shorts']}, {stats['Loser Longs']}, {stats['Loser Shorts']}",
            file=f
        )         

        with open(details_file, 'w') as f:
            transaction_list = []
            for transaction in transactions:
                exit_reason = exit_reasons.get(str(transaction.exit_reason), 'Unknown')
                transaction = {
                    "Entry Index": transaction.entry_index,
                    "Entry Price": float(transaction.entry_price),
                    "Entry Force": float(transaction.entry_force),
                    "Side": int(transaction.side),
                    "Quantity": int(transaction.quantity),
                    "Take Profit": float(transaction.take_profit),
                    "Stop Loss": float(transaction.stop_loss),
                    "PL": float(transaction.pl),
                    "Exit Index": int(transaction.exit_index),
                    "Exit Price": float(transaction.exit_price),
                    "Exit Reason": exit_reason,
                    "position_history": [{"index": s.index, "open_price": float(s.open_price), "high_price": float(s.high_price), "low_price": float(s.low_price), "close_price": float(s.close_price), "dP": float(s.δP), "V": float(s.V), "H": float(s.H), "previous_force": float(s.previous_force), "current_force": float(s.current_force)} for s in transaction.state]
                }
                transaction_list.append(transaction)
            print(json.dumps(transaction_list), file=f)        