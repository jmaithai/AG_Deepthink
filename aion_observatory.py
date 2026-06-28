import pandas as pd
import numpy as np
import os
import warnings

warnings.simplefilter(action='ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_aion_observatory():
    file_path = "archive/fractal_manifold_6M.parquet"
    print(f"[*] AION OBSERVATORY: Loading Raw Sensorium from {file_path}")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Error: {e}")
        return
        
    # ---------------------------------------------------------
    # 1. THE SENSORIUM (Event-Based Clock)
    # ---------------------------------------------------------
    print("[*] Eradicating Chronological Time...")
    # We use M1 tick volume as a proxy for physical network transactions
    vol_cols = [c for c in df.columns if c.endswith('_M1_vol')]
    df['network_ticks'] = df[vol_cols].sum(axis=1)
    df['cumulative_ticks'] = df['network_ticks'].cumsum()
    
    # Event clock ticks every 25,000 physical network transactions
    EVENT_THRESHOLD = 25000 
    df['Event_Clock'] = (df['cumulative_ticks'] // EVENT_THRESHOLD).astype(int)
    
    close_cols = [c for c in df.columns if c.endswith('_M1_close')]
    symbols = [c.replace('_M1_close', '') for c in close_cols]
    
    df['chronological_time'] = df.index
    sensorium = df.groupby('Event_Clock').last()
    print(f"[+] Compressed {len(df):,} chronological minutes into {len(sensorium):,} Event Packets.")
    
    # ---------------------------------------------------------
    # 2. ACTION PHYSICS
    # ---------------------------------------------------------
    print("[*] Calculating Velocity, Acceleration, and Jerk...")
    prices = np.log(sensorium[close_cols].values)
    velocity = np.diff(prices, axis=0, prepend=prices[0:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
    jerk = np.diff(acceleration, axis=0, prepend=acceleration[0:1])
    
    sensorium['System_Jerk'] = np.linalg.norm(jerk, axis=1)
    
    # ---------------------------------------------------------
    # 3. REALITY DEGENERACY & HIDDEN STATE INFERENCE
    # ---------------------------------------------------------
    print("[*] Performing Unsupervised Eigen-Decomposition (Listening to the Data)...")
    
    window = 21 # Rolling event window
    degeneracy_records = []
    
    for i in range(window, len(sensorium)):
        v_window = velocity[i-window:i]
        
        # Add microscopic noise to prevent singular matrices on dead/flat data
        v_noisy = v_window + np.random.normal(0, 1e-12, v_window.shape)
        cov_matrix = np.cov(v_noisy.T)
        
        # Extract Latent Forces (Eigen-Decomposition)
        # This mathematically finds the hidden causes WITHOUT human labels
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by explanatory power (largest eigenvalue first)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        total_energy = np.sum(eigenvalues) + 1e-12
        p_energy = np.clip(eigenvalues / total_energy, 1e-12, 1.0)
        
        # Shannon Entropy (Reality Degeneracy)
        # How many possible explanations are still alive?
        S_degeneracy = -np.sum(p_energy * np.log2(p_energy))
        
        dom_vector = eigenvectors[:, 0]
        dom_power = p_energy[0]
        
        degeneracy_records.append({
            'Event_State': i,
            'Chronological_Time': sensorium['chronological_time'].iloc[i],
            'System_Jerk': sensorium['System_Jerk'].iloc[i],
            'Reality_Degeneracy': S_degeneracy,
            'Dominant_Power': dom_power,
            'Dominant_Vector': dom_vector
        })
        
    results = pd.DataFrame(degeneracy_records)
    
    # Phase Transitions = Rapid Drop in Degeneracy (Wave-Function Collapse)
    results['Degeneracy_Drop'] = results['Reality_Degeneracy'].diff().fillna(0)
    
    print("\n[*] Scanning for Structural Ruptures (Wave-Function Collapses)...")
    
    # We look for the most violent drops in degeneracy (entropy falling)
    # where the market suddenly aligned into a single explanation
    collapses = results[results['Degeneracy_Drop'] < -0.15].sort_values(by='Degeneracy_Drop')
    
    # Filter closely grouped events to isolate distinct days
    collapses['Day'] = collapses['Chronological_Time'].dt.floor('D')
    top_collapses = collapses.drop_duplicates(subset=['Day']).head(5)
    
    print("\n" + "="*100)
    print("AION OBSERVATORY: TOP 5 WAVE-FUNCTION COLLAPSES (REALITY COMPRESSION)")
    print("="*100)
    
    for rank, (_, event) in enumerate(top_collapses.iterrows(), 1):
        ts = event['Chronological_Time'].strftime('%Y-%m-%d %H:%M:%S')
        deg = event['Reality_Degeneracy']
        drop = event['Degeneracy_Drop']
        jrk = event['System_Jerk']
        power = event['Dominant_Power'] * 100
        
        print(f"\n[ PHASE TRANSITION {rank} | CHRONO TIME: {ts} ]")
        print(f"Action Physics:     High Structural Jerk detected ({jrk:.6f})")
        print(f"Reality Degeneracy: Collapsed by {abs(drop):.4f} bits (Current Entropy: {deg:.4f})")
        print(f"Hidden Cause:       A single dominant explanation now commands {power:.2f}% of the manifold.")
        
        # Decode the DOMINANT REALITY
        vec = event['Dominant_Vector']
        loadings = {symbols[j]: vec[j] for j in range(len(symbols))}
        sorted_loadings = sorted(loadings.items(), key=lambda item: abs(item[1]), reverse=True)
        
        print(f"\n  [ PROPAGATION & TRAP MAP ]")
        print(f"  Epicenters (Origin):  [{sorted_loadings[0][0]:<6}] (Weight: {sorted_loadings[0][1]:+.4f}) | [{sorted_loadings[1][0]:<6}] (Weight: {sorted_loadings[1][1]:+.4f})")
        print(f"  Wavefront Receivers:  [{sorted_loadings[2][0]:<6}] (Weight: {sorted_loadings[2][1]:+.4f}) | [{sorted_loadings[3][0]:<6}] (Weight: {sorted_loadings[3][1]:+.4f})")
        
        lag_1, lag_2 = sorted_loadings[-1], sorted_loadings[-2]
        print(f"  Dead Zones / Traps:   [{lag_1[0]:<6}] (Weight: {lag_1[1]:+.4f}) | [{lag_2[0]:<6}] (Weight: {lag_2[1]:+.4f})")
        print(f"  AION Arbiter:         Pressure propagates from Epicenters. Weakest exit is forced through Dead Zones.")

    print("\n" + "="*100)
    print("AION Doctrine Applied. Zero human assumptions. The data defined its own geometry.")

if __name__ == "__main__":
    run_aion_observatory()
