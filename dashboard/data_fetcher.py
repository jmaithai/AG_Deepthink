import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
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
                    
    print(f"Hooked {len(active_symbols)} dimensions. Fetching last 3 days of tick data...")
    
    date_to = datetime.now()
    date_from = date_to - timedelta(days=3)
    
    all_ticks = []
    for sym in active_symbols:
        print(f"Fetching {sym}...")
        ticks = mt5.copy_ticks_range(sym, date_from, date_to, mt5.COPY_TICKS_ALL)
        if ticks is not None and len(ticks) > 0:
            df = pd.DataFrame(ticks)
            df['symbol'] = sym
            df['price'] = (df['bid'] + df['ask']) / 2.0
            all_ticks.append(df[['time_msc', 'symbol', 'price', 'ask', 'bid']])
            
    if all_ticks:
        print("Concatenating and sorting...")
        final_df = pd.concat(all_ticks, ignore_index=True)
        final_df.sort_values(by='time_msc', inplace=True)
        final_df.reset_index(drop=True, inplace=True)
        
        print(f"Total ticks: {len(final_df)}. Saving to ticks_history.pkl...")
        final_df.to_pickle("ticks_history.pkl")
        print("Done.")
    else:
        print("No ticks found.")
        
    mt5.shutdown()

if __name__ == "__main__":
    main()
