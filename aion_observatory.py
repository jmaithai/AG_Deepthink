import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_observatory():
    file_path = "archive/fractal_manifold_6M.parquet"
    print(f"[*] AION OBSERVATORY: Ingesting Fractal Energy Field from {file_path}")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Critical Error: Data missing. {e}")
        return
        
    print("[*] SENSORIUM: Eradicating Chronological Time...")
    vol_cols = [c for c in df.columns if c.endswith('_M1_vol')]
    df['network_ticks'] = df[vol_cols].sum(axis=1)
    df['cumulative_ticks'] = df['network_ticks'].cumsum()
    
    # Event clock ticks based on global network density
    EVENT_THRESHOLD = int(df['network_ticks'].mean() * 1000) 
    df['Event_Clock'] = (df['cumulative_ticks'] // EVENT_THRESHOLD).astype(int)
    
    symbols = [c.replace('_M1_price', '') for c in df.columns if c.endswith('_M1_price')]
    
    # Preserve chronological time for the Living Thesis output
    df['chronological_time'] = df.index
    sensorium = df.groupby('Event_Clock').last()
    
    print(f"[*] Compressed into {len(sensorium):,} True Phase States.")

    print("[*] FIELD CONSTRUCTION: Calculating Macro/Meso/Micro Kinematics...")
    
    p_h1 = sensorium[[f"{sym}_H1_price" for sym in symbols]].values
    p_m15 = sensorium[[f"{sym}_M15_price" for sym in symbols]].values
    p_m1 = sensorium[[f"{sym}_M1_price" for sym in symbols]].values
    
    # Velocities across the fractals
    v_h1 = np.diff(p_h1, axis=0, prepend=p_h1[0:1])
    v_m15 = np.diff(p_m15, axis=0, prepend=p_m15[0:1])
    v_m1 = np.diff(p_m1, axis=0, prepend=p_m1[0:1])
    
    print("[*] REALITY COMPRESSION: Inferring Hidden States & Degeneracy...")
    window = 21
    entropies = np.zeros(len(sensorium))
    
    # We use M15 (Meso) to define Reality Degeneracy, as it captures the propagating wave best
    for i in range(len(sensorium)):
        if i < window:
            entropies[i] = np.nan
            continue
            
        v_window = v_m15[i-window:i]
        
        mean_v = np.mean(v_window, axis=0)
        std_v = np.std(v_window, axis=0) + 1e-12
        v_norm = (v_window - mean_v) / std_v
        
        cov_matrix = np.cov(v_norm.T)
        eig_vals, _ = np.linalg.eigh(cov_matrix)
        
        p_energy = np.clip(eig_vals / (np.sum(eig_vals) + 1e-12), 1e-12, 1.0)
        entropy = -np.sum(p_energy * np.log2(p_energy))
        entropies[i] = entropy
        
    sensorium['Entropy'] = entropies
    sensorium['Entropy_Change'] = sensorium['Entropy'].diff()
    
    print("[*] TRAP DETECTION: Locating Forced Asymmetry...")
    
    # Find the single most violent reality compression (Wave-Function Collapse)
    valid_states = sensorium.dropna(subset=['Entropy_Change'])
    collapse_idx = valid_states['Entropy_Change'].idxmin()
    
    # Locate the exact index mathematically
    i = sensorium.index.get_loc(collapse_idx)
    
    # Extract the dominant vector at the exact moment of collapse
    v_window = v_m15[i-window:i]
    mean_v = np.mean(v_window, axis=0)
    std_v = np.std(v_window, axis=0) + 1e-12
    v_norm = (v_window - mean_v) / std_v
    
    cov_matrix = np.cov(v_norm.T)
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eig_vals)[::-1]
    
    dom_force = eig_vecs[:, idx[0]]
    power = (eig_vals[idx[0]] / np.sum(eig_vals)) * 100
    
    # THE TRAP LOGIC
    # Expected movement based on the Dominant Meso Force
    force_magnitude = np.dot((v_m15[i] - mean_v) / std_v, dom_force)
    expected_v = force_magnitude * dom_force
    
    # Actual Micro (M1) movement
    actual_v_m1 = (v_m1[i] - np.mean(v_m1[i-window:i], axis=0)) / (np.std(v_m1[i-window:i], axis=0) + 1e-12)
    
    # The Trap: Expected to move strongly, but M1 velocity is resisting or dead.
    trap_score = np.abs(expected_v) - np.abs(actual_v_m1)
    
    # We only care about nodes that ARE part of the dominant wave (high expected_v)
    trap_score = np.where(np.abs(expected_v) > np.percentile(np.abs(expected_v), 70), trap_score, -np.inf)
    
    trap_idx = np.argmax(trap_score)
    prop_idx = np.argmax(np.abs(expected_v)) # The Epicenter / Origin
    
    trap_sym = symbols[trap_idx]
    prop_sym = symbols[prop_idx]
    
    ts = sensorium['chronological_time'].iloc[i].strftime('%Y-%m-%d %H:%M:%S')
    
    print("\n" + "="*80)
    print("AION OBSERVATORY: LIVING THESIS")
    print("="*80)
    print(f"EVENT STATE: {collapse_idx} | T=0: {ts}")
    
    print("\n[ WHAT IS THE MARKET DOING? ]")
    print(f"Current Phase State: FRACTURE & COMPRESSION.")
    print(f"Reality Degeneracy collapsed violently by {abs(sensorium['Entropy_Change'].iloc[i]):.4f} bits.")
    
    print("\n[ WHY IS IT DOING IT? ]")
    print(f"Hidden Cause Hypothesis: A singular structural force commands {power:.1f}% of the manifold's energy.")
    print(f"Propagation: The pressure wave originates and is anchored by [{prop_sym}].")
    
    print("\n[ HOW DO WE GAIN FROM IT? ]")
    print(f"Trap Detected: [{trap_sym}] is mathematically compelled by the Meso wave (Eigen-Weight: {np.abs(dom_force[trap_idx]):.4f}).")
    print(f"Physics: The Meso field demands [{trap_sym}] move with trajectory {expected_v[trap_idx]:+.2f}.")
    print(f"         But local Micro (M1) participants are stubbornly holding it at {actual_v_m1[trap_idx]:+.2f}.")
    print(f"         Participants fighting the wave on the micro scale are fundamentally trapped.")
    
    print("\n[ ACTION PHYSICS ARBITER ]")
    direction = "LONG (BUY)" if expected_v[trap_idx] > 0 else "SHORT (SELL)"
    print(f"Weakest Outlet: The forced micro-liquidation of [{trap_sym}] participants.")
    print(f"Action: Execute asymmetric route -> {direction} [{trap_sym}] on the structural snap.")
    print("="*80)

if __name__ == "__main__":
    run_observatory()
