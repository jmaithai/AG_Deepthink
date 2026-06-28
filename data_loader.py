import MetaTrader5 as mt5
import pandas as pd
import os
from datetime import datetime

def fetch_historical_data(symbol, timeframe, start_date, end_date):
    """Downloads MT5 history and saves as a compressed Parquet file."""
    
    if not mt5.initialize():
        print("MT5 Initialization failed.")
        return None

    # Convert strings to datetime
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    print(f"[*] Fetching {symbol} from {start_date} to {end_date}...")
    
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    mt5.shutdown()
    
    if rates is None or len(rates) == 0:
        print("No data retrieved.")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Save to Parquet
    filename = f"data_{symbol}_{timeframe}.parquet"
    df.to_parquet(filename)
    print(f"[+] Saved {len(df)} rows to {filename}")
    return df

if __name__ == "__main__":
    # Example: 6 months of XAUUSD M1 data
    # Change these dates to your desired test window
    fetch_historical_data("XAUUSD", mt5.TIMEFRAME_M1, "2026-01-01", "2026-06-27")
