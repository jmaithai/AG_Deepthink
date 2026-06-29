import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

def test_live_sensorium():
    print("[*] Testing Live MT5 Connection & Manifold Assembly...")
    if not mt5.initialize():
        print("[-] MT5 Init Failed.")
        return
        
    print("[+] MT5 Initialized successfully.")

    valid_symbols = {
        'EURUSD', 'USDCHF', 'USDCAD', 'AUDUSD', 'GBPUSD', 'NZDUSD', 'USDJPY',
        'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'CADCHF', 'CADJPY', 'CHFJPY',
        'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD', 'GBPAUD',
        'GBPCAD', 'GBPJPY', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'NZDJPY',
        'XAUUSD', 'XAGUSD', 'WTI', 'BRENT', 'NATGAS'
    }
    
    print(f"[*] Requesting live M1 data for {len(valid_symbols)} Doctrine nodes...")
    
    dfs = []
    for sym in valid_symbols:
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M1, 0, 100) # Pull last 100 minutes
        if rates is None or len(rates) == 0:
            print(f"[!] Warning: Could not pull data for {sym}. Is it enabled in Market Watch?")
            continue
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df[['close', 'tick_volume']].rename(columns={
            'close': f'{sym}_close',
            'tick_volume': f'{sym}_vol'
        })
        dfs.append(df)
        
    if not dfs:
        print("[-] Fatal: No data pulled. Check MT5 connection and symbols.")
        mt5.shutdown()
        return
        
    print("[*] Assembling Topological Manifold...")
    manifold = pd.concat(dfs, axis=1).fillna(method='ffill').dropna()
    
    print(f"[+] Live Manifold assembled. Shape: {manifold.shape}")
    
    vol_cols = [c for c in manifold.columns if c.endswith('_vol')]
    manifold['network_ticks'] = manifold[vol_cols].sum(axis=1)
    manifold['cumulative_ticks'] = manifold['network_ticks'].cumsum()
    
    current_ticks = manifold['cumulative_ticks'].iloc[-1]
    
    print("\n==================================================")
    print("LIVE SENSORIUM DIAGNOSTICS:")
    print("==================================================")
    print(f"Active Nodes Pulled: {len(vol_cols)}")
    print(f"Total Network Ticks in window: {current_ticks:,.0f}")
    print(f"Average Ticks per Minute: {manifold['network_ticks'].mean():,.0f}")
    print(f"Time required for 1 True Event State (25k ticks): {(25000 / manifold['network_ticks'].mean()):.1f} chronological minutes")
    print("==================================================")
    
    print("\n[ VERDICT ] LIVE CONNECTION TEST PASSED. SENSORIUM IS ACTIVE.")
    
    mt5.shutdown()

if __name__ == "__main__":
    test_live_sensorium()
