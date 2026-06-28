import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def fetch_omni_energy_stream():
    print("[*] Initializing OMNI Unsupervised Energy Ingestor...")
    if not mt5.initialize():
        print("[-] MT5 Init Failed.")
        return

    # 1. Unsupervised Topology Discovery
    # Pull every single symbol the broker offers. No human lists.
    symbols = mt5.symbols_get()
    if symbols is None:
        print("[-] Failed to retrieve symbols.")
        return
        
    print(f"[*] Discovered {len(symbols)} total assets in the broker universe.")
    
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=48) # 48-hour continuous tick stream
    
    raw_streams = []
    active_symbols = []
    
    # Known crypto base currencies to exclude
    CRYPTO_BASES = {'BTC','ETH','LTC','XRP','BCH','ADA','DOT','LINK','BNB','SOL',
                    'DOGE','MATIC','AVAX','UNI','ATOM','XLM','VET','TRX','EOS','XMR'}

    print(f"[*] Fetching continuous tick stream and filtering for liquid nodes...")

    for sym in symbols:
        name = sym.name

        # FILTER 1: Skip broker-disabled symbols (trade_mode == 0)
        if sym.trade_mode == 0:
            continue

        # FILTER 2: Skip crypto — check path folder OR base/profit currency
        sym_path = sym.path.upper()
        if 'CRYPTO' in sym_path:
            continue
        if sym.currency_base.upper() in CRYPTO_BASES:
            continue
        if sym.currency_profit.upper() in CRYPTO_BASES:
            continue

        if not sym.visible:
            mt5.symbol_select(name, True)

        ticks = mt5.copy_ticks_range(name, start_time, end_time, mt5.COPY_TICKS_ALL)
        
        if ticks is None or len(ticks) < 10000:
            # Filter: If it has less than 10,000 ticks in 48 hours, it is a dead node (illiquid).
            continue
            
        df = pd.DataFrame(ticks)
        df['time'] = pd.to_datetime(df['time_msc'], unit='ms')
        df.set_index('time', inplace=True)
        
        # Deduplicate same-millisecond ticks
        temp_df = pd.DataFrame(index=df.index)
        temp_df[f"{name}_mid"] = (df['bid'] + df['ask']) / 2.0
        temp_df[f"{name}_vol"] = df['volume']
        temp_df = temp_df[~temp_df.index.duplicated(keep='last')]
        
        raw_streams.append(temp_df)
        active_symbols.append(name)
        
        if len(active_symbols) % 5 == 0:
            print(f" [+] Validated {len(active_symbols)} liquid nodes so far (Latest: {name})...")

    mt5.shutdown()

    if not raw_streams:
        print("[-] Fatal: No tick streams retrieved.")
        return

    print(f"\n[*] Fusing {len(active_symbols)}-Dimensional continuous event cloud...")
    
    master_df = pd.concat(raw_streams, axis=1, join='outer')
    
    price_cols = [c for c in master_df.columns if '_mid' in c]
    vol_cols = [c for c in master_df.columns if '_vol' in c]
    
    master_df[price_cols] = master_df[price_cols].ffill()
    master_df[vol_cols] = master_df[vol_cols].fillna(0)
    master_df.dropna(subset=price_cols, inplace=True)
    
    os.makedirs("archive", exist_ok=True)
    filename = "archive/omni_energy_cloud.parquet"
    master_df.to_parquet(filename)
    
    print(f"\n[SUCCESS] OMNI Event Cloud saved: {filename}")
    print(f"[*] Total Validated Nodes (The True Manifold): {len(active_symbols)}")
    print(f"[*] Liquid Symbols: {', '.join(active_symbols)}")
    print(f"[*] Total Discrete System Events: {len(master_df):,}")

if __name__ == "__main__":
    fetch_omni_energy_stream()
