import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os
import argparse
import json
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--months', type=int, default=6)
    args, _ = parser.parse_known_args()
    
    TERMINAL_PATH = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = json.load(f)
            TERMINAL_PATH = cfg.get('TERMINAL_PATH', None)
            
    init_kwargs = {}
    if TERMINAL_PATH:
        init_kwargs['path'] = TERMINAL_PATH
        
    if not mt5.initialize(**init_kwargs):
        print(f"MT5 init failed. Path: {TERMINAL_PATH}")
        return
        
    active_symbols = []
    all_symbols = mt5.symbols_get()
    if all_symbols:
        for sym in all_symbols:
            if sym.visible and sym.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                p = sym.path.lower()
                if any(k in p for k in ['forex', 'fx', 'metal', 'energi', 'commod']):
                    active_symbols.append(sym.name)
                    
    print(f"[*] Hooked {len(active_symbols)} dimensions. Fetching last {args.months} months of tick data...")
    
    os.makedirs("tick_data", exist_ok=True)
    
    date_to = datetime.now()
    date_from_absolute = date_to - timedelta(days=30 * args.months)
    
    current_end = date_to
    chunk_index = 0
    
    while current_end > date_from_absolute:
        current_start = current_end - timedelta(days=7)
        if current_start < date_from_absolute:
            current_start = date_from_absolute
            
        print(f"\n[*] Fetching Chunk {chunk_index}: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
        
        chunk_ticks = []
        for sym in active_symbols:
            # print(f"  Fetching {sym}...", end="\r")
            ticks = mt5.copy_ticks_range(sym, current_start, current_end, mt5.COPY_TICKS_ALL)
            if ticks is not None and len(ticks) > 0:
                df = pd.DataFrame(ticks)
                df['symbol'] = sym
                df['price'] = (df['bid'] + df['ask']) / 2.0
                chunk_ticks.append(df[['time_msc', 'symbol', 'price', 'ask', 'bid']])
                
        if chunk_ticks:
            final_df = pd.concat(chunk_ticks, ignore_index=True)
            final_df.sort_values(by='time_msc', inplace=True)
            final_df.reset_index(drop=True, inplace=True)
            
            chunk_file = f"tick_data/chunk_{chunk_index:03d}.parquet"
            final_df.to_parquet(chunk_file)
            print(f"  [+] Saved {len(final_df)} ticks to {chunk_file}")
        else:
            print(f"  [-] No ticks found for this chunk.")
            
        current_end = current_start
        chunk_index += 1
        
    print("\n[*] Data Fetching Complete.")
    mt5.shutdown()

if __name__ == "__main__":
    main()
