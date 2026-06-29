import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def solve_the_problem():
    # Gauge Inversion Fix: Path matches actual parquet
    file_path = "archive/fractal_manifold_6M.parquet"
    print("[*] AION CORE: Ingesting Omni-Dimensional Energy Field...")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Error: {e}")
        return

    # 1. BARLESS SENSORIUM
    print("[*] Eradicating Chronological Time...")
    # Gauge Inversion Fix: _M1_vol
    vol_cols = [c for c in df.columns if c.endswith('_M1_vol')]
    df['network_ticks'] = df[vol_cols].sum(axis=1)
    df['cumulative_ticks'] = df['network_ticks'].cumsum()
    
    # Event clock ticks scale dynamically to network density
    EVENT_THRESHOLD = int(df['network_ticks'].mean() * 1000) 
    df['Event_Clock'] = (df['cumulative_ticks'] // EVENT_THRESHOLD).astype(int)
    
    # Gauge Inversion Fix: _M15_price (macro tide)
    close_cols = [c for c in df.columns if c.endswith('_M15_price')]
    symbols = [c.replace('_M15_price', '') for c in close_cols]
    
    df['chronological_time'] = df.index
    sensorium = df.groupby('Event_Clock').last()
    
    print(f"[*] Compressed into {len(sensorium):,} pure thermodynamic states.")

    # 2. THE FIELD (Velocity & Covariance)
    # Gauge Inversion Fix: Already log-transformed
    prices = sensorium[close_cols].values
    velocity = np.diff(prices, axis=0, prepend=prices[0:1])

    print("[*] Scanning for Dynamic Manifold Formations and Tension Traps...")
    
    window = 21
    prev_entropy = 0
    
    for i in range(window, len(sensorium)):
        v_window = velocity[i-window:i]
        
        # Normalize to find pure structural shape
        mean_v = np.mean(v_window, axis=0)
        std_v = np.std(v_window, axis=0) + 1e-12
        v_norm = (v_window[-1] - mean_v) / std_v # Instantaneous state
        
        v_window_norm = (v_window - mean_v) / std_v
        cov_matrix = np.cov(v_window_norm.T)
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eig_vals)[::-1]
        
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]
        
        total_energy = np.sum(eig_vals) + 1e-12
        p_energy = np.clip(eig_vals / total_energy, 1e-12, 1.0)
        entropy = -np.sum(p_energy * np.log2(p_energy))
        
        if i > window:
            entropy_drop = entropy - prev_entropy
            
            # 3. TRIGGER: Phase Transition (Wave-Function Collapse)
            if entropy_drop < -0.15:
                
                # 4. SOLVING THE WHAT, WHY, AND HOW (No Pigeonholing)
                # How many independent manifolds are currently active?
                # We ask the data. Any force holding > 15% of the global energy is an active manifold.
                active_manifolds = np.where(p_energy > 0.15)[0]
                
                if len(active_manifolds) > 0:
                    ts = sensorium['chronological_time'].iloc[i].strftime('%Y-%m-%d %H:%M:%S')
                    
                    print("\n" + "="*85)
                    print(f"AION LIVING THESIS  |  T=0: {ts}")
                    print("="*85)
                    print(f"[ WHAT IS THE MARKET DOING? ]")
                    print(f"Phase State: FRACTURE & COMPRESSION.")
                    print(f"Reality Degeneracy collapsed by {abs(entropy_drop):.4f} bits. (Current Entropy: {entropy:.2f})")
                    print(f"Emergent Topology: The data has self-organized into {len(active_manifolds)} distinct, active macro-forces.\n")
                    
                    # Analyze each independent force separately
                    for m_idx in active_manifolds:
                        dom_force = eig_vecs[:, m_idx]
                        power = p_energy[m_idx] * 100
                        
                        force_magnitude = np.dot(v_norm, dom_force)
                        expected_v = force_magnitude * dom_force
                        actual_v = np.abs(v_norm) + 1e-9 
                        
                        # THE HOW: Tension Discrepancy
                        # Theoretical pull of the force divided by actual kinetic movement
                        z_trap = np.abs(expected_v) / actual_v
                        
                        trap_idx = np.argmax(z_trap)
                        prop_idx = np.argmax(np.abs(expected_v)) # Epicenter
                        
                        trap_sym = symbols[trap_idx]
                        prop_sym = symbols[prop_idx]
                        direction = "LONG (BUY)" if expected_v[trap_idx] > 0 else "SHORT (SELL)"
                        
                        print(f"--- FORCE FIELD {m_idx + 1} (Commanding {power:.1f}% of System Energy) ---")
                        print(f"    [ WHY ] Propagation Anchor: [{prop_sym}] is driving this dimension.")
                        print(f"    [ HOW ] Trap Detected:      [{trap_sym}] (Tension Discrepancy: {z_trap[trap_idx]:.2f}x)")
                        print(f"            Action Arbiter:     EXECUTE {direction} [{trap_sym}] to capture structural snap.\n")
                    
                    print("="*85)
                    break # Show the first major chronological rupture to prove the solution.
                
        prev_entropy = entropy

if __name__ == "__main__":
    solve_the_problem()
