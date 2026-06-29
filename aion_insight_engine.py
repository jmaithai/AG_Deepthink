import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_insight_engine():
    # Gauge Inversion Fixes
    file_path = "archive/fractal_manifold_6M.parquet"
    print("[*] AION INSIGHT ENGINE: Mapping the Living Energy Web...")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Error: {e}")
        return

    # 1. BARLESS SENSORIUM (Event Clock)
    print("[*] Eradicating Chronological Time...")
    vol_cols = [c for c in df.columns if c.endswith('_M1_vol')]
    df['network_ticks'] = df[vol_cols].sum(axis=1)
    df['cumulative_ticks'] = df['network_ticks'].cumsum()
    
    EVENT_THRESHOLD = int(df['network_ticks'].mean() * 1000) 
    df['Event_Clock'] = (df['cumulative_ticks'] // EVENT_THRESHOLD).astype(int)
    
    close_cols = [c for c in df.columns if c.endswith('_M15_price')]
    symbols = np.array([c.replace('_M15_price', '') for c in close_cols])
    
    df['chronological_time'] = df.index
    sensorium = df.groupby('Event_Clock').last()
    
    print(f"[*] Compressed into {len(sensorium):,} pure thermodynamic states.")

    # 2. STATE VECTORS
    raw_prices = sensorium[close_cols].values
    # Prices are already log-transformed in the Parquet tensor
    velocity = np.diff(raw_prices, axis=0, prepend=raw_prices[0:1])
    
    print("[*] Constructing Graph Topology & Mapping Energy Diffusion...")
    
    window = 50
    found_insight = False
    
    for i in range(window, len(sensorium)):
        v_window = velocity[i-window:i]
        
        # The Silk Threads (Correlations)
        corr_matrix = np.corrcoef(v_window.T)
        np.fill_diagonal(corr_matrix, 0)
        
        # Energy Injection (Who is moving right now?)
        current_v = velocity[i]
        
        # Network Propagation (The Wave)
        # Expected velocity based on the flow from all other nodes via correlation threads
        expected_v = np.dot(corr_matrix, current_v)
        
        # 3. INSIGHT GENERATION (The Dam)
        # We look for a node where the Network is pushing hard (High Expected), 
        # but the node is frozen or moving in reverse (Low or Opposing Actual)
        
        discrepancy = np.abs(expected_v - current_v)
        
        # Filter: We only care if the network wave is exceptionally strong
        if np.max(np.abs(expected_v)) > np.percentile(np.abs(velocity[i-window:i]), 95):
            
            trap_idx = np.argmax(discrepancy)
            
            # Ensure it's a real structural trap (Network demands a move 3x larger than actual)
            if np.abs(expected_v[trap_idx]) > (np.abs(current_v[trap_idx]) * 3.0):
                
                trap_sym = symbols[trap_idx]
                
                # Identify the Origin: Who is pushing the web?
                contributions = corr_matrix[trap_idx] * current_v
                puller_idx = np.argmax(np.abs(contributions))
                puller_sym = symbols[puller_idx]
                
                ts = sensorium['chronological_time'].iloc[i].strftime('%Y-%m-%d %H:%M:%S')
                
                print("\n" + "="*85)
                print(f"AION ENERGY INSIGHT  |  T=0: {ts}")
                print("="*85)
                
                print("[ WHAT IS HAPPENING? (THE WAVE) ]")
                print(f"Kinetic energy is actively diffusing through the network topology.")
                print(f"A massive wave originating at [{puller_sym}] is propagating across the correlation threads.")
                
                print("\n[ WHY IS IT HAPPENING? (THE DAM) ]")
                print(f"The web topology identifies [{trap_sym}] as the primary structural bottleneck.")
                print(f"The network is exerting a physical propagating force of {expected_v[trap_idx]:+.4f} on [{trap_sym}],")
                print(f"but its actual kinetic velocity is only {current_v[trap_idx]:+.4f}.")
                print(f"Local participants are absorbing the macro shockwave, building a dam.")
                
                print("\n[ WHAT IS MOST LIKELY TO HAPPEN? (THE ALPHA) ]")
                direction = "UPWARD" if expected_v[trap_idx] > 0 else "DOWNWARD"
                print(f"Conservation of energy mandates diffusion. The dam at [{trap_sym}] cannot hold indefinitely.")
                print(f"Expect a violent kinetic catch-up {direction} as the node is ripped into alignment.")
                print("="*85)
                found_insight = True
                break

    if not found_insight:
        print("[-] The network is currently flowing freely. No structural dams detected.")

if __name__ == "__main__":
    run_insight_engine()
