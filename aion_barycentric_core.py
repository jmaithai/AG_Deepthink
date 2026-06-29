import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_barycentric_audit():
    file_path = "archive/fractal_manifold_6M.parquet"
    print(f"[*] AION HAMILTONIAN ENGINE: Mapping True Physics (Barycentric Model)...")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Error: {e}")
        return
        
    print("[*] Applying Doctrine Firewall: Filtering non-tradable nodes...")
    valid_symbols = {
        'EURUSD', 'USDCHF', 'USDCAD', 'AUDUSD', 'GBPUSD', 'NZDUSD', 'USDJPY',
        'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'CADCHF', 'CADJPY', 'CHFJPY',
        'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD', 'GBPAUD',
        'GBPCAD', 'GBPJPY', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'NZDJPY',
        'XAUUSD', 'XAGUSD', 'WTI', 'BRENT', 'NATGAS'
    }
    
    close_cols = [c for c in df.columns if c.endswith('_M1_close') and c.replace('_M1_close', '') in valid_symbols]
    vol_cols = [c.replace('_M1_close', '_M1_vol') for c in close_cols]
    
    print("[*] Eradicating Chronological Time...")
    df['network_ticks'] = df[vol_cols].sum(axis=1)
    
    # Doctrine Firewall: Fallback if dataset lacks volume
    if df['network_ticks'].sum() == 0:
        df['network_ticks'] = 1.0
        
    df['cumulative_ticks'] = df['network_ticks'].cumsum()
    
    # Event clock ticks based on network density, not human clocks
    EVENT_THRESHOLD = 25000 
    df['Event_Clock'] = (df['cumulative_ticks'] // EVENT_THRESHOLD).astype(int)
    
    symbols = [c.replace('_M1_close', '') for c in close_cols]
    
    df['chronological_time'] = df.index
    sensorium = df.groupby('Event_Clock').last()
    
    print(f"[+] Compressed into {len(sensorium):,} True Event States.")

    # 1. State Vectors (Normalized for cross-asset spatial geometry)
    raw_prices = sensorium[close_cols].values
    prices = (raw_prices - np.mean(raw_prices, axis=0)) / (np.std(raw_prices, axis=0) + 1e-9)
    
    volumes = sensorium[vol_cols].values
    # Mass is normalized volume. We offset to ensure strictly positive mass for physics calculations.
    mass = (volumes - np.mean(volumes, axis=0)) / (np.std(volumes, axis=0) + 1e-9)
    mass = np.clip(mass + 1.0, 0.1, None) 
    
    velocity = np.diff(prices, axis=0, prepend=prices[0:1])
    
    print("[*] Calculating Manifold Barycenter and Energy States...")
    
    # 2. THE MANIFOLD BARYCENTER (The True Equilibrium)
    # Volume-weighted geometric center of the N-Dimensional manifold at every tick
    barycenter = np.sum(mass * prices, axis=1) / np.sum(mass, axis=1)
    
    # 3. HAMILTONIAN MECHANICS
    T = 0.5 * mass * (velocity**2) # Kinetic Energy
    displacement = prices - barycenter[:, None]
    V = 0.5 * mass * (displacement**2) # Potential Energy
    
    H = T + V # Total System Energy
    delta_H = np.diff(H, axis=0, prepend=H[0:1])

    print("[*] Auditing The Law of the Coiled Spring...")
    
    passes = 0
    failures = 0
    
    # Analyze the manifold
    for i in range(10, len(sensorium) - 20):
        
        current_V = V[i]
        current_T = T[i]
        current_dH = delta_H[i]
        
        v_thresh = np.percentile(current_V, 90)
        t_thresh = np.percentile(current_T, 10)
        
        # The Trap: Highly stretched (V), frozen (T), and actively absorbing energy (dH)
        is_coiled = (current_V > v_thresh) & (current_T < t_thresh) & (current_dH > 0)
        
        if is_coiled.any():
            look_forward = 20
            for idx in np.where(is_coiled)[0]:
                disp_t0 = abs(displacement[i, idx])
                
                future_disp = np.abs(displacement[i+1 : i+1+look_forward, idx])
                future_T = T[i+1 : i+1+look_forward, idx]
                
                min_future_disp = np.min(future_disp)
                max_future_T = np.max(future_T)
                
                # FALSIFICATION: Did it physically snap back toward the Barycenter AND release kinetic energy?
                if min_future_disp < disp_t0 and max_future_T > current_T[idx]:
                    passes += 1
                else:
                    failures += 1

    win_rate = (passes / (passes + failures)) * 100 if (passes + failures) > 0 else 0
    
    print("\n" + "="*80)
    print("AION HAMILTONIAN (BARYCENTRIC) AUDIT: RESULTS")
    print("="*80)
    print(f"Total Coiled Traps Identified: {passes + failures}")
    print(f"Mathematical Win Rate (Forced Release): {win_rate:.2f}%")
    print("="*80)
    if win_rate > 50:
        print("[ VERDICT ] POSITIVE CONVEXITY DETECTED. THE PHYSICS HOLD.")
    else:
        print("[ VERDICT ] DOCTRINE FAILED. NOISE DETECTED.")

if __name__ == "__main__":
    run_barycentric_audit()
