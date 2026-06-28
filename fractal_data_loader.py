import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import physics_engine as pe
import os

def fetch_fractal_manifold():
    print("[*] Initializing MT5 for Fractal Extraction...")
    if not mt5.initialize():
        print("[-] MT5 Init Failed.")
        return

    nodes, edges = pe.discover_topology()
    if not edges:
        return

    end_date = datetime.now()
    start_date = end_date - timedelta(days=180) # 6 Months
    
    # The Fractal Scales: Micro, Meso, Macro
    timeframes = {
        'M1': mt5.TIMEFRAME_M1,
        'M15': mt5.TIMEFRAME_M15,
        'H1': mt5.TIMEFRAME_H1
    }

    df_list = []
    missing = []
    
    for tf_name, tf_val in timeframes.items():
        print(f"\n[*] Fetching Frequency: {tf_name}")
        for edge in edges:
            sym = edge['symbol']
            rates = mt5.copy_rates_range(sym, tf_val, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                print(f" [!] Missing {tf_name} data for {sym}. Please cache it in MT5.")
                missing.append(f"{sym} ({tf_name})")
                continue
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            temp_df = pd.DataFrame(index=df.index)
            # Log price for Gauge Inversion (The Physics)
            temp_df[f"{sym}_{tf_name}_price"] = np.log(df['close'])
            # Volume for Mass Calculation
            temp_df[f"{sym}_{tf_name}_vol"] = df['tick_volume']
            # Raw price for TRUE, unleveraged PnL tracking
            temp_df[f"{sym}_{tf_name}_close"] = df['close']
            
            df_list.append(temp_df)
            print(f" [+] {sym} ({tf_name}): {len(df)} bars retrieved.")

    mt5.shutdown()

    if missing:
        print(f"\n[!] {len(missing)} missing datasets:")
        for m in missing:
            print(f"    -> {m}")
        print("[!] Open these charts in MT5 and hold 'Page Up', then re-run.")

    if not df_list:
        print("[-] Fatal: No data retrieved.")
        return

    print("\n[*] Fusing multi-dimensional tensor (Harmonic Alignment)...")
    # Concat aligns timestamps. Forward-fill allows the H1/M15 macro states 
    # to persist inside the M1 micro-ticks until the next macro bar closes.
    master_df = pd.concat(df_list, axis=1)
    master_df.ffill(inplace=True)
    master_df.dropna(inplace=True)
    
    os.makedirs("archive", exist_ok=True)
    filename = "archive/fractal_manifold_6M.parquet"
    master_df.to_parquet(filename)
    print(f"\n[SUCCESS] Fractal Manifold saved: {filename}")
    print(f"[*] Tensor Shape: {master_df.shape[0]} rows x {master_df.shape[1]} columns")

if __name__ == "__main__":
    fetch_fractal_manifold()
