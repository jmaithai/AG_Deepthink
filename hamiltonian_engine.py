import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_hamiltonian_engine():
    file_path = "archive/manifold_6M.parquet"
    print(f"[*] AION HAMILTONIAN ENGINE: Mapping True Physics (6M OOS Dataset)...")
    
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
    
    close_cols = [c for c in df.columns if c.endswith('_close') and c.replace('_close', '') in valid_symbols]
    vol_cols = [c.replace('_close', '_vol') for c in close_cols]
    
    print("[*] Eradicating Chronological Time...")
    df['network_ticks'] = df[vol_cols].sum(axis=1)
    
    # Fallback to absolute ticks if volume is unrecorded
    if df['network_ticks'].sum() == 0:
        df['network_ticks'] = 1.0
        
    df['cumulative_ticks'] = df['network_ticks'].cumsum()
    
    # Event clock ticks based on pure network density, not human clocks
    # 50,000 cumulative network ticks = 1 True Event State
    EVENT_THRESHOLD = 50000
    df['Event_Clock'] = (df['cumulative_ticks'] // EVENT_THRESHOLD).astype(int)
    
    symbols = [c.replace('_close', '') for c in close_cols]
    
    df['chronological_time'] = df.index
    sensorium = df.groupby('Event_Clock').last()
    
    print(f"[+] Compressed into {len(sensorium):,} True Event States.")

    # 1. State Vectors (Normalized for cross-asset spatial geometry)
    raw_prices = sensorium[close_cols].values
    prices = (raw_prices - np.mean(raw_prices, axis=0)) / (np.std(raw_prices, axis=0) + 1e-9)
    
    volumes = sensorium[vol_cols].values
    mass = (volumes - np.mean(volumes, axis=0)) / (np.std(volumes, axis=0) + 1e-9)
    mass = np.clip(mass, 0.1, None) # Mass must be strictly positive
    
    velocity = np.diff(prices, axis=0, prepend=prices[0:1])
    
    print("[*] Calculating Manifold Barycenter and Energy States...")
    
    # 2. THE MANIFOLD BARYCENTER (The True Equilibrium)
    # Volume-weighted center of mass of the entire N-Dimensional manifold
    barycenter = np.sum(mass * prices, axis=1) / np.sum(mass, axis=1)
    
    # 3. HAMILTONIAN MECHANICS
    # Kinetic Energy (T) = 0.5 * m * v^2
    T = 0.5 * mass * (velocity**2)
    
    # Potential Energy (V) = 0.5 * m * d^2 (Distance from Barycenter)
    displacement = prices - barycenter[:, None]
    V = 0.5 * mass * (displacement**2)
    
    # Total Energy (H) & Flow (Delta H)
    H = T + V
    delta_H = np.diff(H, axis=0, prepend=H[0:1])

    print("[*] Auditing The Law of the Coiled Spring...")
    
    passes = 0
    failures = 0
    
    # Analyze the manifold
    for i in range(10, len(sensorium) - 10):
        # Define a "Coil" locally relative to the manifold's current state
        # High V (top 10% of manifold), Low T (bottom 10% of manifold), absorbing energy (dH > 0)
        
        current_V = V[i]
        current_T = T[i]
        current_dH = delta_H[i]
        
        v_thresh = np.percentile(current_V, 90)
        t_thresh = np.percentile(current_T, 10)
        
        # The Trap: Highly stretched, frozen, and actively absorbing energy
        is_coiled = (current_V > v_thresh) & (current_T < t_thresh) & (current_dH > 0)
        
        if is_coiled.any():
            # Check the next 10 Event States (Physical time, not chronological)
            # Did the coiled node release its potential energy and snap back?
            
            for idx in np.where(is_coiled)[0]:
                disp_t0 = abs(displacement[i, idx])
                # Displacement 10 energy states in the future
                disp_tf = abs(displacement[i+10, idx]) 
                
                # Check for kinetic spike
                max_v_future = np.max(np.abs(velocity[i+1:i+11, idx]))
                
                # FALSIFICATION: Did it snap back toward the Barycenter AND release kinetic energy?
                if disp_tf < disp_t0 and max_v_future > current_T[idx]:
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
    run_hamiltonian_engine()
