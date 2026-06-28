import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def fetch_raw_energy_stream():
    print("[*] Initializing Unsupervised Energy Ingestor...")
    if not mt5.initialize():
        print("[-] MT5 Init Failed.")
        return

    # We select the highest volume nodes to ensure a continuous stream of events
    # We do not assume their importance; we only select them for data density.
    target_edges = [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", 
        "USDCHF", "NZDUSD", "XAUUSD", "EURJPY", "GBPJPY"
    ]

    end_time = datetime.now()
    # We only pull 48 hours to manage the immense data size of raw ticks
    start_time = end_time - timedelta(hours=48)
    
    print(f"[*] Fetching continuous tick stream from {start_time} to {end_time}...")

    raw_streams = []
    
    for sym in target_edges:
        print(f" [+] Extracting raw event stream for {sym}...")
        # Pulling actual ticks (Bid/Ask/Volume), NOT time-aggregated bars
        ticks = mt5.copy_ticks_range(sym, start_time, end_time, mt5.COPY_TICKS_ALL)
        
        if ticks is None or len(ticks) == 0:
            print(f" [!] Missing tick data for {sym}.")
            continue
            
        df = pd.DataFrame(ticks)
        df['time'] = pd.to_datetime(df['time_msc'], unit='ms') # Sub-millisecond precision
        df.set_index('time', inplace=True)
        
        # We only care about the mid-price (the exact center of the spread) and the volume
        temp_df = pd.DataFrame(index=df.index)
        temp_df[f"{sym}_mid"] = (df['bid'] + df['ask']) / 2.0
        temp_df[f"{sym}_vol"] = df['volume']
        
        # Deduplicate: multiple ticks can share the same millisecond timestamp.
        # Keep the last price event per millisecond to ensure a unique index.
        temp_df = temp_df[~temp_df.index.duplicated(keep='last')]
        raw_streams.append(temp_df)
        print(f"     -> {len(df)} discrete events captured.")

    mt5.shutdown()

    if not raw_streams:
        print("[-] Fatal: No tick stream retrieved.")
        return

    print("\n[*] Fusing multi-dimensional continuous event cloud...")
    # This is a massive outer join. It creates a continuous timeline where every 
    # single sub-millisecond event across any pair is logged.
    # We forward-fill the mid-prices, but fill volume with 0 (since volume is a discrete event)
    master_df = pd.concat(raw_streams, axis=1, join='outer')
    
    price_cols = [c for c in master_df.columns if '_mid' in c]
    vol_cols = [c for c in master_df.columns if '_vol' in c]
    
    master_df[price_cols] = master_df[price_cols].ffill()
    master_df[vol_cols] = master_df[vol_cols].fillna(0)
    
    # Drop rows until all pairs have established a baseline price
    master_df.dropna(subset=price_cols, inplace=True)
    
    os.makedirs("archive", exist_ok=True)
    filename = "archive/raw_energy_cloud.parquet"
    master_df.to_parquet(filename)
    print(f"\n[SUCCESS] Raw Unsupervised Event Cloud saved: {filename}")
    print(f"[*] Total Discrete System Events: {len(master_df)}")

if __name__ == "__main__":
    fetch_raw_energy_stream()
