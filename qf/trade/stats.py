
import numpy as np
   
def get_returns(equity_array):
    if len(equity_array) > 1:
        return np.diff(equity_array) / equity_array[:-1]
    else:
        return np.array([0])

def create_backtest_stats(results):
    # cash, longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts, transaction_log
    (ticker, equity_curve, long_trades, short_trades, winner_longs, winner_shorts, loser_longs, loser_shorts, _) = results
    
    equity_array = np.ravel(equity_curve)
    prev_equity = equity_array[:-1]    

    # Initialize returns with 0
    returns = np.zeros_like(prev_equity)
    
    # Only calculate for indices where capital was still above zero
    valid_mask = prev_equity > 0
    returns[valid_mask] = np.diff(equity_array)[valid_mask] / prev_equity[valid_mask]
    
    volatility = np.std(returns) if len(returns) > 0 else 0
    # If final_capital was an array due to the previous bug, take the first value
    f_cap = float(np.ravel(equity_curve)[0]) if np.ndim(equity_curve) > 0 else equity_curve
    
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
        "Ticker": ticker,
        "Initial Capital": initial_capital,
        "Final Capital": equity_curve[-1],
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
    return stats