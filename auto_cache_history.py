import MetaTrader5 as mt5
import time

# Core symbols for the AION Prime / Fleet Commander systems
# Dynamically resolved after mt5.initialize()
TARGET_SYMBOLS = []

import argparse
import json
import os

def force_cache_m1_history():
    global TARGET_SYMBOLS
    print("[*] Initiating Autonomous M1 History Caching Sequence...")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args, _ = parser.parse_known_args()
    
    init_kwargs = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = json.load(f)
            if 'TERMINAL_PATH' in cfg:
                init_kwargs['path'] = cfg['TERMINAL_PATH']
                
    if not mt5.initialize(**init_kwargs):
        print("[-] mt5.initialize() failed. Ensure MT5 is running.")
        return

    all_symbols = mt5.symbols_get()
    if all_symbols:
        for sym in all_symbols:
            if sym.visible and sym.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                p = sym.path.lower()
                if any(k in p for k in ['forex', 'fx', 'metal', 'energi', 'commod']):
                    TARGET_SYMBOLS.append(sym.name)
    print(f"[*] Dynamically resolved {len(TARGET_SYMBOLS)} symbols for caching.")


    total_cached = 0
    for sym in TARGET_SYMBOLS:
        print(f"[*] Requesting deep M1 history for [{sym}] to force broker download...")
        
        # Requesting 200,000 M1 bars (~6-8 months) programmatically forces 
        # the MT5 terminal to pull the historical data from the broker's server,
        # completely eliminating the need to manually press "Page Up" on the charts.
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M1, 0, 200000)
        
        if rates is None or len(rates) == 0:
            print(f"    [-] Failed to cache {sym}. Data may be unavailable on broker.")
        else:
            print(f"    [+] Successfully pulled and cached {len(rates):,} M1 bars for {sym}")
            total_cached += len(rates)
            
        time.sleep(1.5)  # Throttle to prevent overwhelming the broker's data server

    print(f"\n[+] Caching sequence complete. {total_cached:,} total bars loaded.")
    print("[+] Terminal memory is fully primed for 32-D Kinetic Layering.")
    mt5.shutdown()

if __name__ == "__main__":
    force_cache_m1_history()
