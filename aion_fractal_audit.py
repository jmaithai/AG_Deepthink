import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_crucible_audit():
    file_path = "archive/fractal_manifold_6M.parquet"
    print("[*] AION CRUCIBLE: Initiating 6-Month Statistical Stress Test...")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Error: {e}")
        return

    # Prepare Event Clock
    print("[*] Eradicating Chronological Time...")
    vol_cols = [c for c in df.columns if c.endswith('_M1_vol')]
    df['network_ticks'] = df[vol_cols].sum(axis=1)
    df['cumulative_ticks'] = df['network_ticks'].cumsum()
    EVENT_THRESHOLD = 25000 
    df['Event_Clock'] = (df['cumulative_ticks'] // EVENT_THRESHOLD).astype(int)
    
    # Gauge Inversion Fix
    close_cols_m1 = [c for c in df.columns if c.endswith('_M1_price')]
    close_cols_m15 = [c for c in df.columns if c.endswith('_M15_price')]
    symbols = [c.replace('_M1_price', '') for c in close_cols_m1]
    
    df['chronological_time'] = df.index
    sensorium = df.groupby('Event_Clock').last()
    
    # Kinematics
    prices_m15 = sensorium[close_cols_m15].values
    velocity_m15 = np.diff(prices_m15, axis=0, prepend=prices_m15[0:1])

    print(f"[*] Scanning {len(sensorium):,} Event States for Topological Traps...")
    
    window = 21
    prev_entropy = 0
    trap_events = []
    
    for i in range(window, len(sensorium)):
        v_window = velocity_m15[i-window:i]
        
        mean_v = np.mean(v_window, axis=0)
        std_v = np.std(v_window, axis=0) + 1e-12
        v_norm = (v_window[-1] - mean_v) / std_v 
        
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
            
            # TRIGGER: Reality Degeneracy + High Dimension Energy (>50%)
            if entropy_drop < -0.15 and p_energy[0] > 0.50:
                dom_force = eig_vecs[:, 0]
                
                force_magnitude = np.dot(v_norm, dom_force)
                expected_v = force_magnitude * dom_force
                actual_v = np.abs(v_norm) + 1e-9
                
                # THE HOW: Tension Discrepancy
                z_trap = np.abs(expected_v) / actual_v
                trap_idx = np.argmax(z_trap)
                
                # CRUCIBLE CONDITION
                if z_trap[trap_idx] > 2.5:
                    trap_sym = symbols[trap_idx]
                    direction = 1 if expected_v[trap_idx] > 0 else -1
                    
                    trap_events.append({
                        'Time': sensorium['chronological_time'].iloc[i],
                        'Trap_Sym': trap_sym,
                        'Direction': direction
                    })
                    
        prev_entropy = entropy

    traps_df = pd.DataFrame(trap_events)
    if traps_df.empty:
        print("[-] No events found.")
        return
        
    # Deduplicate events within the same hour for the same symbol
    traps_df['HourGroup'] = traps_df['Time'].dt.floor('H')
    distinct_traps = traps_df.drop_duplicates(subset=['HourGroup', 'Trap_Sym'])
    
    print(f"[+] Discovered {len(distinct_traps)} Topological Traps. Initiating O(log N) Energy Walk...")

    passes = 0
    failures = 0
    release_log = []
    loss_log = []
    
    raw_df = df.copy()
    raw_df['global_energy'] = raw_df[vol_cols].sum(axis=1).cumsum()
    
    for _, trap in distinct_traps.iterrows():
        t0 = trap['Time']
        sym = trap['Trap_Sym']
        direction = trap['Direction']
        
        # O(1) start index lookup
        try:
            start_idx = raw_df.index.get_loc(t0)
            if isinstance(start_idx, slice):
                start_idx = start_idx.start
            elif isinstance(start_idx, np.ndarray):
                start_idx = np.where(start_idx)[0][0]
        except (KeyError, IndexError):
            continue
            
        if start_idx + 5 >= len(raw_df):
            continue
            
        e0 = raw_df['global_energy'].iloc[start_idx]
        e_target = e0 + 100000
        
        # O(log N) end index lookup via binary search on monotonic energy
        end_idx = np.searchsorted(raw_df['global_energy'].values[start_idx:], e_target) + start_idx
        
        forward_df = raw_df.iloc[start_idx:end_idx]
        
        if len(forward_df) < 5: continue
            
        p0 = forward_df[f"{sym}_M1_price"].iloc[0]
        disp = (forward_df[f"{sym}_M1_price"] - p0) * 100
        
        max_up = disp.max()
        max_down = disp.min()
        
        if direction > 0: # LONG
            release = max_up
            friction = abs(max_down)
        else: # SHORT
            release = abs(max_down)
            friction = max_up

        # Falsification logic: Release must overpower friction by 2x
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
    print("AION CRUCIBLE: 6-MONTH STATISTICAL AUDIT")
    print("="*80)
    print(f"Total Traps Identified:    {passes + failures}")
    print(f"Mathematical Win Rate:     {win_rate:.2f}%")
    print(f"Average Kinetic Release:   {avg_release:.4f}% (Yield)")
    print(f"Average Adverse Friction:  {avg_loss:.4f}% (Drawdown)")
    print(f"System Expectancy Ratio:   {expectancy_ratio:.2f}x")
    print(f"Expected Value (EV):       {ev:+.5f}% per event")
    print("="*80)

if __name__ == "__main__":
    run_crucible_audit()
