import pandas as pd
import numpy as np
import os
import warnings

warnings.simplefilter(action='ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_omni_observatory():
    file_path = "archive/omni_energy_cloud.parquet"
    print(f"[*] AION OMNI-OBSERVATORY: Loading 68-D Sensorium from {file_path}")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Error: {e}")
        return
        
    print("[*] Eradicating Chronological Time...")
    # Raw tick data has volume=0/1 per row — use row index as the event clock
    # Each packet = 10,000 discrete sub-millisecond market events
    EVENT_THRESHOLD = 10000
    df['Event_Clock'] = np.arange(len(df)) // EVENT_THRESHOLD
    
    close_cols = [c for c in df.columns if c.endswith('_mid')]
    symbols = [c.replace('_mid', '') for c in close_cols]
    
    df['chronological_time'] = df.index
    sensorium = df.groupby('Event_Clock').last()
    print(f"[+] Compressed {len(df):,} chronological ticks into {len(sensorium):,} Omni-Event Packets.")
    
    print("[*] Calculating Kinematic Deformation (Velocity, Acceleration, Jerk)...")
    prices = np.log(sensorium[close_cols].values + 1e-9)
    velocity = np.diff(prices, axis=0, prepend=prices[0:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
    jerk = np.diff(acceleration, axis=0, prepend=acceleration[0:1])
    
    sensorium['System_Jerk'] = np.linalg.norm(jerk, axis=1)
    
    print("[*] Performing Unsupervised Eigen-Decomposition on 68 Dimensions...")
    
    window = 21 
    degeneracy_records = []
    
    for i in range(window, len(sensorium)):
        v_window = velocity[i-window:i]
        
        # Z-Score normalization per symbol in the rolling window (Correlation Matrix logic)
        # This prevents high-volatility indices (like SPX500, VIX) from artificially drowning out FX
        mean_v = np.mean(v_window, axis=0)
        std_v = np.std(v_window, axis=0) + 1e-12
        v_norm = (v_window - mean_v) / std_v
        
        v_noisy = v_norm + np.random.normal(0, 1e-12, v_norm.shape)
        cov_matrix = np.cov(v_noisy.T)
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        total_energy = np.sum(eigenvalues) + 1e-12
        p_energy = np.clip(eigenvalues / total_energy, 1e-12, 1.0)
        
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
    results['Degeneracy_Drop'] = results['Reality_Degeneracy'].diff().fillna(0)
    
    print("\n[*] Scanning for Structural Ruptures (Wave-Function Collapses)...")
    collapses = results[results['Degeneracy_Drop'] < -0.15].sort_values(by='Degeneracy_Drop')
    
    # Because this is a 48 hour capture, we group by hour to get distinct macro events
    collapses['Hour'] = collapses['Chronological_Time'].dt.floor('H')
    top_collapses = collapses.drop_duplicates(subset=['Hour']).head(5)
    
    print("\n" + "="*110)
    print("AION OMNI-OBSERVATORY: TOP 5 WAVE-FUNCTION COLLAPSES (OMNI MANIFOLD - 48H)")
    print("="*110)
    
    for rank, (_, event) in enumerate(top_collapses.iterrows(), 1):
        ts = event['Chronological_Time'].strftime('%Y-%m-%d %H:%M:%S')
        deg = event['Reality_Degeneracy']
        drop = event['Degeneracy_Drop']
        jrk = event['System_Jerk']
        power = event['Dominant_Power'] * 100
        
        print(f"\n[ PHASE TRANSITION {rank} | CHRONO TIME: {ts} ]")
        print(f"Action Physics:     High Structural Jerk detected ({jrk:.6f})")
        print(f"Reality Degeneracy: Collapsed by {abs(drop):.4f} bits (Current Entropy: {deg:.4f})")
        print(f"Hidden Cause:       A single dominant explanation commands {power:.2f}% of the 68-D manifold.")
        
        vec = event['Dominant_Vector']
        loadings = {symbols[j]: vec[j] for j in range(len(symbols))}
        sorted_loadings = sorted(loadings.items(), key=lambda item: abs(item[1]), reverse=True)
        
        print(f"\n  [ PROPAGATION & TRAP MAP ]")
        print(f"  Epicenters (Origin):  [{sorted_loadings[0][0]:<8}] ({sorted_loadings[0][1]:+.4f}) | [{sorted_loadings[1][0]:<8}] ({sorted_loadings[1][1]:+.4f})")
        print(f"  Wavefront Receivers:  [{sorted_loadings[2][0]:<8}] ({sorted_loadings[2][1]:+.4f}) | [{sorted_loadings[3][0]:<8}] ({sorted_loadings[3][1]:+.4f})")
        
        lag_1, lag_2 = sorted_loadings[-1], sorted_loadings[-2]
        print(f"  Dead Zones / Traps:   [{lag_1[0]:<8}] ({lag_1[1]:+.4f}) | [{lag_2[0]:<8}] ({lag_2[1]:+.4f})")

    print("\n" + "="*110)
    print("AION Omni-Doctrine Applied. The 68-Dimensional structure has mapped itself.")

if __name__ == "__main__":
    run_omni_observatory()
