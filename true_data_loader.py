import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import physics_engine as pe
import os

def fetch_full_manifold_history():
    print("[*] Initializing MT5 for Deep History Extraction...")
    if not mt5.initialize():
        print("[-] MT5 Init Failed.")
        return

    nodes, edges = pe.discover_topology()
    if not edges:
        print("[-] No edges found. Cannot build manifold.")
        mt5.shutdown()
        return

    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    print(f"[*] Fetching 6 months of M1 data ({start_date.date()} to {end_date.date()})...")

    df_list = []
    
    for edge in edges:
        sym = edge['symbol']
        rates = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M1, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            print(f" [!] Missing data for {sym}. Open its M1 chart in MT5 and hold 'Page Up' to cache it.")
            continue
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        temp_df = pd.DataFrame(index=df.index)
        temp_df[f"{sym}_price"] = np.log(df['close'])
        temp_df[f"{sym}_vol"] = df['tick_volume']
        temp_df[f"{sym}_close"] = df['close'] # Need real raw price for accurate PnL
        
        df_list.append(temp_df)
        print(f" [+] {sym}: {len(df)} bars retrieved.")

    mt5.shutdown()

    if not df_list:
        print("[-] FATAL: No historical data retrieved across any pairs.")
        return

    print("[*] Synchronizing multi-dimensional tensor...")
    master_df = pd.concat(df_list, axis=1)
    master_df.ffill(inplace=True)
    master_df.dropna(inplace=True)
    
    os.makedirs("archive", exist_ok=True)
    filename = "archive/manifold_6M.parquet"
    master_df.to_parquet(filename)
    print(f"\n[SUCCESS] True Manifold History saved: {filename}")
    print(f"[+] Shape: {master_df.shape[0]} rows x {master_df.shape[1]} columns")

if __name__ == "__main__":
    fetch_full_manifold_history()
