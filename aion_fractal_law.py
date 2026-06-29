import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def extract_frequency_physics(df, active_edges, tf_label):
    cols = [f"{e['symbol']}_{tf_label}_price" for e in active_edges]
    sub_df = df[cols].copy()
    # Gauge Inversion Fix: Data is already log-transformed
    prices = sub_df.values
    velocity = np.diff(prices, axis=0, prepend=prices[0:1])
    return velocity

def run_fractal_law():
    file_path = "archive/fractal_manifold_6M.parquet"
    print(f"[*] AION FRACTAL LAW ENGINE: Loading {file_path}")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Error: {e}")
        return
        
    print("[*] Eradicating Chronological Time...")
    vol_cols = [c for c in df.columns if c.endswith('_M1_vol')]
    df['network_ticks'] = df[vol_cols].sum(axis=1)
    df['cumulative_ticks'] = df['network_ticks'].cumsum()
    EVENT_THRESHOLD = int(df['network_ticks'].mean() * 1000) 
    df['Event_Clock'] = (df['cumulative_ticks'] // EVENT_THRESHOLD).astype(int)
    
    # Gauge Inversion Fix: The column suffix is _M1_price, not _M1_close
    close_cols = [c for c in df.columns if c.endswith('_M1_price')]
    symbols = [c.replace('_M1_price', '') for c in close_cols]
    active_edges = [{'symbol': sym} for sym in symbols]
    
    raw_df = df.copy()
    raw_df['chronological_time'] = raw_df.index
    
    # In pandas, grouping by Event_Clock preserves columns that are not the group key.
    # We must explicitly ensure 'chronological_time' is preserved as a datetime column.
    df['chronological_time'] = df.index
    sensorium = df.groupby('Event_Clock').last()
    
    print("[*] Extracting Meso (M15) and Micro (M1) Kinematics...")
    v_m15 = extract_frequency_physics(sensorium, active_edges, "M15")
    v_m1 = extract_frequency_physics(sensorium, active_edges, "M1")
    
    print("[*] Scanning 6 Months for Wave-Function Collapses...")
    window = 21
    prev_entropy = 0
    ruptures = []
    
    for i in range(window, len(sensorium)):
        v_window = v_m15[i-window:i]
        mean_v = np.mean(v_window, axis=0)
        std_v = np.std(v_window, axis=0) + 1e-12
        v_norm = (v_window - mean_v) / std_v
        
        cov_matrix = np.cov(v_norm.T)
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[idx]
        
        p_energy = np.clip(eig_vals / (np.sum(eig_vals) + 1e-12), 1e-12, 1.0)
        entropy = -np.sum(p_energy * np.log2(p_energy))
        
        if i > window:
            entropy_drop = entropy - prev_entropy
            
            # TRIGGER: Violent collapse in Meso Reality Degeneracy
            if entropy_drop < -0.15 and p_energy[0] > 0.40:
                dom_force = eig_vecs[:, idx[0]]
                
                # THE TRAP LOGIC
                force_magnitude = np.dot((v_m15[i] - mean_v) / std_v, dom_force)
                expected_v = force_magnitude * dom_force
                
                actual_v_m1 = (v_m1[i] - np.mean(v_m1[i-window:i], axis=0)) / (np.std(v_m1[i-window:i], axis=0) + 1e-12)
                
                trap_score = np.abs(expected_v) - np.abs(actual_v_m1)
                trap_score = np.where(np.abs(expected_v) > np.percentile(np.abs(expected_v), 70), trap_score, -np.inf)
                
                trap_idx = np.argmax(trap_score)
                trap_sym = symbols[trap_idx]
                direction = -1 if expected_v[trap_idx] < 0 else 1 # -1 for Short, 1 for Long
                
                ruptures.append({
                    'Time': sensorium['chronological_time'].iloc[i],
                    'Trap': trap_sym,
                    'Direction': direction
                })
                
        prev_entropy = entropy

    ruptures_df = pd.DataFrame(ruptures)
    if ruptures_df.empty: return
    
    ruptures_df['HourGroup'] = ruptures_df['Time'].dt.floor('H')
    distinct_ruptures = ruptures_df.drop_duplicates(subset=['HourGroup', 'Trap'])
    
    print(f"[+] Discovered {len(distinct_ruptures)} valid Fractal Traps.")
    print("[*] Initiating Timeless Forward-Walk Falsification (100,000 tick energy injection)...")

    passes = 0
    failures = 0
    release_log = []
    loss_log = []
    
    for _, rupture in distinct_ruptures.iterrows():
        t0 = rupture['Time']
        trap_sym = rupture['Trap']
        direction = rupture['Direction']
        
        post_rupture_df = raw_df[raw_df.index >= t0].copy()
        if len(post_rupture_df) < 5: continue
            
        vol_cols = [c for c in post_rupture_df.columns if c.endswith('_M1_vol')]
        post_rupture_df['post_energy'] = post_rupture_df[vol_cols].sum(axis=1).cumsum()
        forward_df = post_rupture_df[post_rupture_df['post_energy'] <= 100000]
        
        if len(forward_df) < 5: continue
            
        p0 = forward_df[f"{trap_sym}_M1_price"].iloc[0]
        # Gauge Inversion Fix: Math for displacement is (log(Pt) - log(P0)) since values are already log()
        disp = (forward_df[f"{trap_sym}_M1_price"] - p0) * 100
        
        max_up = disp.max()
        max_down = disp.min()
        
        # FALSIFICATION LOGIC
        if direction > 0: # LONG
            release = max_up
            friction = abs(max_down)
        else: # SHORT
            release = abs(max_down)
            friction = max_up

        # The release must overpower the friction by 2x
        if release > (friction * 2) and release > 0.05: 
            passes += 1
            release_log.append(release)
        else:
            failures += 1
            loss_log.append(friction)

    win_rate = (passes / (passes + failures)) * 100 if (passes + failures) > 0 else 0
    avg_release = np.mean(release_log) if release_log else 0.0
    avg_loss = np.mean(loss_log) if loss_log else 0.0
    expectancy_ratio = avg_release / (avg_loss + 0.0001) if passes > 0 else 0.0
    ev = (win_rate/100 * avg_release) - ((1 - win_rate/100) * avg_loss)

    print("\n" + "="*80)
    print("AION FRACTAL LAW ENGINE: 6-MONTH STATISTICAL AUDIT")
    print("="*80)
    print(f"Total Traps Identified:    {passes + failures}")
    print(f"Mathematical Win Rate:     {win_rate:.2f}%")
    print(f"Average Kinetic Release:   {avg_release:.4f}% (Yield)")
    print(f"Average Adverse Friction:  {avg_loss:.4f}% (Drawdown)")
    print(f"System Expectancy Ratio:   {expectancy_ratio:.2f}x")
    print(f"Expected Value (EV):       {ev:+.5f}% per event")
    print("="*80)

if __name__ == "__main__":
    run_fractal_law()
