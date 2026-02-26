import sys
import os
import json
import pandas as pd
from sqlalchemy import create_engine
from qf.dbsync import db_config
import mplfinance as mpf
import matplotlib.pyplot as plt

if __name__ == "__main__":
    quote_name = sys.argv[1]
    test_results_dir = os.path.join(os.getcwd(), "test-results")
    transactions_file = os.path.join(test_results_dir, f"report-{quote_name}-transactions.json")
    _, sqlalchemy_url = db_config()
    engine = create_engine(sqlalchemy_url)
    losers = 0
    winners = 0

    # Loads transactions log
    with open(transactions_file, 'r') as f:
        transactions = json.load(f)
    
    # Calculate transactions range
    min_entry_date =  min([t['Entry Date'] for t in transactions])[:10]
    max_exit_date = max([t['Exit Date'] for t in transactions])[:10]

    # Loads OHLC bars encompased within transactions range
    with engine.connect() as connection:
        df = pd.read_sql(f"SELECT \"TIMESTAMP\", \"OPEN\", \"HIGH\", \"LOW\", \"CLOSE\" FROM \"QUOTE\" WHERE \"TICKER\" = '{quote_name}' AND \"TIMESTAMP\"::DATE BETWEEN '{min_entry_date}' AND '{max_exit_date}' ORDER BY \"TIMESTAMP\"", connection) 
        connection.close()    
    engine.dispose()

    # Plot candle sticks from df
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    df.set_index('TIMESTAMP', inplace=True)
    df.rename(columns={'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low', 'CLOSE': 'Close'}, inplace=True)

# 1. Initialize lists for markers (filled with NaN)
    # We use NaN so that markers only appear where we have data
    long_entries = [float('nan')] * len(df)
    short_entries = [float('nan')] * len(df)
    winner_exits = [float('nan')] * len(df)
    loser_exits = [float('nan')] * len(df)

    # 2. Iterate over transactions to populate the marker lists
    for t in transactions:
        # Parse dates
        entry_date = pd.to_datetime(t['Entry Date'])
        exit_date = pd.to_datetime(t['Exit Date'])
        
        # Determine position side (1 = Long, -1 = Short)
        side = t['Side']
        
        # Add Entry Marker
        if entry_date in df.index:
            # Find the numeric index (row number) for this date
            idx = df.index.get_loc(entry_date)
            high_price = df.loc[entry_date]['High']
            low_price = df.loc[entry_date]['Low']
            
            if side == 1:
                # Long Entry: Blue Triangle Up below the low
                long_entries[idx] = low_price * 0.99 
            else:
                # Short Entry: Red Triangle Down above the high
                short_entries[idx] = high_price * 1.01

        # Add Exit Marker
        if exit_date in df.index:
            idx = df.index.get_loc(exit_date)
            high_price = df.loc[exit_date]['High']
            
            if t['PL'] > 0:
                winner_exits[idx] = high_price * 1.02
                winners += 1
            else:
                loser_exits[idx] = high_price * 1.02
                losers += 1

    # 3. Create addplots for the markers
    ap_long = mpf.make_addplot(long_entries, type='scatter', markersize=100, marker='^', color='blue', label='↑')
    ap_short = mpf.make_addplot(short_entries, type='scatter', markersize=100, marker='v', color='black', label='↓')
    ap_winner_exit = mpf.make_addplot(winner_exits, type='scatter', markersize=80, marker='x', color='green', label='✓')
    ap_loser_exit = mpf.make_addplot(loser_exits, type='scatter', markersize=80, marker='x', color='red', label='X')

    # 4. Plot the candlestick chart with the markers

    addplot = []
    if len(ap_long) > 0:
        addplot.append(ap_long)
    if len(ap_short) > 0:
        addplot.append(ap_short)
    if winners > 0:
        addplot.append(ap_winner_exit)
    if losers > 0:
        addplot.append(ap_loser_exit)

    mpf.plot(
        df, 
        type='candle', 
        style='yahoo', 
        title=f'Price Action for {quote_name}', 
        ylabel='Price',
        volume=False, 
        show_nontrading=False,
        addplot = addplot
    )