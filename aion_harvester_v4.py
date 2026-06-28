import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import warnings

# AION DOCTRINE: Systemic Resonance Harvester
# Detecting the capacitor state (Locked Outlet) during manifold phase transition.

warnings.filterwarnings('ignore')
CRYPTO_FILTER = {'UNIUSD', 'SUIUSD', 'BTCUSD', 'ETHUSD', 'SOLUSD', 'LINKUSD', 'ADAUSD', 'XAGUSD', 'XAUUSD'}

def run_harvester():
    if not mt5.initialize():
        print("[-] MT5 Init Failed.")
        return

    # Ingest: Purified Manifold (No crypto noise)
    all_symbols = mt5.symbols_get()
    symbols = [sym.name for sym in all_symbols if sym.visible and sym.name not in CRYPTO_FILTER]
    
    print(f"[*] AION HARVESTER (v4): Monitoring {len(symbols)} nodes.")
    print(f"[*] Trigger: Entropy < 3.5 AND Impedance Ratio Z < 0.3")
    print(f"[*] Detecting Resonance...")
    print(f"[*] Press Ctrl+C to halt.\n")
    
    # Live buffer: 1000 ticks for short-term manifold state
    buffer_len = 1000
    data_buffer = pd.DataFrame(np.nan, index=range(buffer_len), columns=symbols, dtype=float)

    tick_count = 0
    
    while True:
        # 1. Live Sensorium
        for sym in symbols:
            tick = mt5.symbol_info_tick(sym)
            if tick and tick.bid > 0 and tick.ask > 0:
                data_buffer.loc[buffer_len-1, sym] = (tick.bid + tick.ask) / 2.0
        
        # Shift and clean
        data_buffer = data_buffer.shift(-1).ffill().bfill()

        tick_count += 1
        if tick_count < 50:
            print(f"  [~] Warming up buffer... ({tick_count}/50)", end='\r')
            time.sleep(0.5)
            continue
        
        # 2. Physics: Kinematic Deformation (Normalized Velocity)
        prices = np.log(data_buffer.values.astype(float) + 1e-9)
        velocity = np.diff(prices, axis=0)
        norm_vel = (velocity - np.mean(velocity, axis=0)) / (np.std(velocity, axis=0) + 1e-12)
        
        # 3. Eigen-Decomposition (The Hidden Field)
        cov_matrix = np.cov(norm_vel.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        
        # 4. The Eureka Physics: Coupling Constant (Z) — Impedance Ratio
        dom_vec = eigenvectors[:, idx[0]]
        epi_idx = int(np.argmax(np.abs(dom_vec)))
        out_idx = int(np.argmin(np.abs(dom_vec)))
        
        epi_vol = np.std(norm_vel[:, epi_idx])
        out_vol = np.std(norm_vel[:, out_idx])
        Z = out_vol / (epi_vol + 1e-9)
        
        # 5. Reality Degeneracy (Shannon Entropy)
        total = np.sum(np.abs(eigenvalues)) + 1e-12
        p_energy = np.abs(eigenvalues[idx]) / total
        p_energy = np.clip(p_energy, 1e-12, 1.0)
        entropy = -np.sum(p_energy * np.log2(p_energy))
        
        # 6. TRIGGER: Systemic Phase Transition + Locked Capacitor
        # Condition: Manifold ordered (S < 3.5) AND Outlet pinned (Z < 0.3)
        if entropy < 3.5 and Z < 0.3:
            print(f"\n{'='*60}")
            print(f"[!] AION RESONANCE DETECTED — LOCKED CAPACITOR STATE")
            print(f"{'='*60}")
            print(f"    Entropy:        {entropy:.4f}  (Threshold: 3.5)")
            print(f"    Impedance (Z):  {Z:.4f}   (Threshold: 0.3)")
            print(f"    Epicenter:      {symbols[epi_idx]} (Velocity: {epi_vol:.4f})")
            print(f"    Forced Outlet:  {symbols[out_idx]} (Z-Impedance: {Z:.4f})")
            print(f"    Action: Execute forced-release trade on [{symbols[out_idx]}].")
            print(f"    NULL_AVOID: If [{symbols[out_idx]}] fails to move within 4 Event States, abort.")
            print(f"{'='*60}\n")
        else:
            if tick_count % 60 == 0:
                print(f"  [~] Tick #{tick_count} | S={entropy:.4f} | Z={Z:.4f} | Watching...", end='\r')
            
        time.sleep(0.5)

if __name__ == "__main__":
    run_harvester()
