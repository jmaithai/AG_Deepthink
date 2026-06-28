import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# GLOBAL PURGE LIST
CRYPTO_FILTER = {'UNIUSD', 'SUIUSD', 'BTCUSD', 'ETHUSD', 'SOLUSD', 'LINKUSD', 'ADAUSD'}

def run_harvester():
    if not mt5.initialize():
        print("[-] MT5 Init Failed.")
        return

    # Filtered Ingestion
    all_symbols = mt5.symbols_get()
    symbols = [sym.name for sym in all_symbols if sym.visible and sym.name not in CRYPTO_FILTER]
    
    print(f"[*] AION HARVESTER: Monitoring {len(symbols)} purified nodes...")
    print(f"[*] Trigger: Entropy < 3.5 AND Dominant Power > 50%")
    print(f"[*] Press Ctrl+C to halt.\n")
    
    # Live buffer (1000 ticks)
    data_buffer = pd.DataFrame(np.nan, index=range(1000), columns=symbols, dtype=float)
    
    tick_count = 0

    while True:
        # 1. Live Sensorium
        for sym in symbols:
            tick = mt5.symbol_info_tick(sym)
            if tick and tick.bid > 0 and tick.ask > 0:
                data_buffer.loc[999, sym] = (tick.bid + tick.ask) / 2.0
        
        data_buffer = data_buffer.shift(-1).ffill().bfill()

        tick_count += 1
        if tick_count < 50:
            print(f"  [~] Warming up buffer... ({tick_count}/50)", end='\r')
            time.sleep(0.5)
            continue

        # 2. Physics: Kinematic Deformation
        prices = np.log(data_buffer.values.astype(float) + 1e-9)
        velocity = np.diff(prices, axis=0)
        
        # 3. Reality Degeneracy
        # Normalize to prevent outliers (Indices) from dominating
        mean_v = np.mean(velocity, axis=0)
        std_v = np.std(velocity, axis=0) + 1e-12
        v_norm = (velocity - mean_v) / std_v
        
        cov_matrix = np.cov(v_norm.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        
        # Energy distribution
        total = np.sum(np.abs(eigenvalues)) + 1e-12
        p_energy = np.abs(eigenvalues[idx]) / total
        entropy = -np.sum(p_energy * np.log2(p_energy + 1e-12))
        
        # 4. Trigger Logic: Only fire on Systemic Collapse
        # Entropy must be low (system ordered) AND dominant eigenvector > 50% variance
        if entropy < 3.5 and p_energy[0] > 0.50:
            vec = eigenvectors[:, idx[0]]
            loadings = {symbols[j]: vec[j] for j in range(len(symbols))}
            sorted_loadings = sorted(loadings.items(), key=lambda item: abs(item[1]), reverse=True)
            
            print(f"\n{'='*60}")
            print(f"[!] AION SYSTEMIC SNAP DETECTED")
            print(f"{'='*60}")
            print(f"    Dominant Power: {p_energy[0]*100:.1f}%")
            print(f"    Entropy:        {entropy:.4f}")
            print(f"    Epicenter:      {sorted_loadings[0][0]}")
            print(f"    Receiver:       {sorted_loadings[1][0]}")
            print(f"    Forced Outlet:  {sorted_loadings[-1][0]}")
            print(f"    Action: Monitor [{sorted_loadings[-1][0]}] for kinetic release.")
            print(f"{'='*60}\n")
        else:
            if tick_count % 60 == 0:
                print(f"  [~] Tick #{tick_count} | Entropy: {entropy:.4f} | Dom: {p_energy[0]*100:.1f}% | Watching...", end='\r')
            
        time.sleep(0.5)

if __name__ == "__main__":
    run_harvester()
