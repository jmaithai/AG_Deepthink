import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_historical_aion_core():
    file_path = "archive/omni_energy_cloud.parquet"
    print("[*] AION CORE HISTORICAL TEST: Ingesting Omni-Dimensional Energy Field...")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Error: {e}")
        return

    # Use the safe liquid tradable list to prevent false exotic collapses
    valid_symbols = {
        'EURUSD', 'USDCHF', 'USDCAD', 'AUDUSD', 'GBPUSD', 'NZDUSD', 'USDJPY',
        'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'CADCHF', 'CADJPY', 'CHFJPY',
        'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD', 'GBPAUD',
        'GBPCAD', 'GBPJPY', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'NZDJPY',
        'XAUUSD', 'XAGUSD', 'WTI', 'BRENT', 'NATGAS'
    }
    
    close_cols = [c for c in df.columns if c.endswith('_mid') and c.replace('_mid', '') in valid_symbols]
    vol_cols = [c.replace('_mid', '_vol') for c in close_cols]
    symbols = [c.replace('_mid', '') for c in close_cols]
    
    print(f"[*] Fusing historical tensor with {len(symbols)} liquid dimensions...")
    
    print("\n[*] AION CORE SOLVER: Eradicating Chronological Time...")
    df['network_ticks'] = df[vol_cols].sum(axis=1)
    
    if df['network_ticks'].sum() == 0:
        df['network_ticks'] = 1.0 # fallback for zero volume
        
    df['cumulative_ticks'] = df['network_ticks'].cumsum()
    
    # Event Threshold scales dynamically to network density
    # 5000 ticks per state (as we did previously for this dataset)
    EVENT_THRESHOLD = max(1000, int(df['network_ticks'].mean() * 5000)) 
    df['Event_Clock'] = (df['cumulative_ticks'] // EVENT_THRESHOLD).fillna(0).astype(int)
    
    df['chronological_time'] = df.index
    sensorium = df.groupby('Event_Clock').last()
    
    print(f"[*] Compressed into {len(sensorium):,} pure thermodynamic states.")

    # THE FIELD (Velocity & Covariance)
    prices = np.log(sensorium[close_cols].values + 1e-9)
    velocity = np.diff(prices, axis=0, prepend=prices[0:1])

    print("[*] Scanning Manifold for Reality Degeneracy and Tension Traps...")
    
    window = 21
    prev_entropy = 0
    found_rupture = False
    
    for i in range(window, len(sensorium)):
        v_window = velocity[i-window:i]
        
        # Normalize to find pure structural shape
        mean_v = np.mean(v_window, axis=0)
        std_v = np.std(v_window, axis=0) + 1e-12
        v_norm = (v_window[-1] - mean_v) / std_v # Current state normalized
        
        v_window_norm = (v_window - mean_v) / std_v
        cov_matrix = np.cov(v_window_norm.T)
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eig_vals)[::-1]
        
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]
        
        # Reality Degeneracy (Entropy)
        total_energy = np.sum(eig_vals) + 1e-12
        p_energy = np.clip(eig_vals / total_energy, 1e-12, 1.0)
        entropy = -np.sum(p_energy * np.log2(p_energy))
        
        # 3. PHASE TRANSITION TRIGGER
        if i > window:
            entropy_drop = entropy - prev_entropy
            
            # Trigger: Entropy collapses AND a single force dominates
            if entropy_drop < -0.20 and p_energy[0] > 0.40:
                
                dom_force = eig_vecs[:, 0]
                power = p_energy[0] * 100
                
                # How hard is the Manifold pushing right now?
                force_magnitude = np.dot(v_norm, dom_force)
                
                # What the Manifold DEMANDS each node to do
                expected_v = force_magnitude * dom_force
                
                # THE TRAP (Tension Discrepancy = Expected - Actual)
                tension_discrepancy = expected_v - v_norm
                
                trap_idx = np.argmax(np.abs(tension_discrepancy))
                prop_idx = np.argmax(np.abs(expected_v)) # Epicenter
                
                trap_sym = symbols[trap_idx]
                prop_sym = symbols[prop_idx]
                
                ts = sensorium['chronological_time'].iloc[i].strftime('%Y-%m-%d %H:%M:%S')
                
                print("\n" + "="*85)
                print(f"AION LIVING THESIS  |  T=0: {ts}")
                print("="*85)
                
                print("[ WHAT IS THE MARKET DOING? ]")
                print(f"Phase State: FRACTURE & COMPRESSION.")
                print(f"Reality Degeneracy collapsed by {abs(entropy_drop):.4f} bits. (Current Entropy: {entropy:.2f})")
                
                print("\n[ WHY IS IT DOING IT? ]")
                print(f"Hidden Cause: A singular structural force commands {power:.1f}% of the {len(symbols)}-D manifold.")
                print(f"Propagation: The pressure wave is visibly anchored by [{prop_sym}].")
                
                print("\n[ HOW DO WE GAIN FROM IT? ]")
                trap_exp = expected_v[trap_idx]
                trap_act = v_norm[trap_idx]
                
                print(f"Trap Detected: [{trap_sym}] has massive Tension Discrepancy.")
                print(f"Physics: The {len(symbols)}-D Manifold demands [{trap_sym}] move with velocity {trap_exp:+.2f}.")
                print(f"         But local participants are holding it at {trap_act:+.2f}.")
                print(f"         Retail is fighting the global Eigen-force. They are fatally trapped.")
                
                print("\n[ ACTION PHYSICS ARBITER ]")
                direction = "LONG (BUY)" if trap_exp > 0 else "SHORT (SELL)"
                print(f"Weakest Outlet: The forced liquidation of [{trap_sym}] participants.")
                print(f"Verdict: TRAP IDENTIFIED. EXECUTE FORCED RELEASE -> {direction} [{trap_sym}].")
                print("="*85)
                
                found_rupture = True
                break # Show first major rupture
                
        prev_entropy = entropy
        
    if not found_rupture:
        print("\n[-] Manifold is currently stable. No structural ruptures detected in the historical timeframe.")

if __name__ == "__main__":
    run_historical_aion_core()
