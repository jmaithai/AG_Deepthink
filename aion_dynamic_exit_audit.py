import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_dynamic_exit_audit():
    file_path = "archive/fractal_manifold_6M.parquet"
    print("[*] AION DYNAMIC EXIT AUDIT: Eradicating Retail Falsification Constraints...")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Error: {e}")
        return

    print("[*] Eradicating Chronological Time...")
    vol_cols = [c for c in df.columns if c.endswith('_M1_vol')]
    df['network_ticks'] = df[vol_cols].sum(axis=1)
    df['cumulative_ticks'] = df['network_ticks'].cumsum()
    EVENT_THRESHOLD = 25000 
    df['Event_Clock'] = (df['cumulative_ticks'] // EVENT_THRESHOLD).astype(int)
    
    close_cols_m1 = [c for c in df.columns if c.endswith('_M1_price')]
    close_cols_m15 = [c for c in df.columns if c.endswith('_M15_price')]
    symbols = [c.replace('_M1_price', '') for c in close_cols_m1]
    
    df['chronological_time'] = df.index
    sensorium = df.groupby('Event_Clock').last()
    
    prices_m15 = sensorium[close_cols_m15].values
    velocity_m15 = np.diff(prices_m15, axis=0, prepend=prices_m15[0:1])

    print(f"[*] Scanning {len(sensorium):,} Event States for Topological Traps...")
    
    window = 21
    prev_entropy = 0
    
    passes = 0
    failures = 0
    release_log = []
    loss_log = []
    
    i = window
    while i < len(sensorium):
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
            relative_drop = entropy_drop / (prev_entropy + 1e-6)
            
            if relative_drop < -0.15 and p_energy[0] > 0.40:
                dom_force = eig_vecs[:, 0]
                
                force_magnitude = np.dot(v_norm, dom_force)
                expected_v = force_magnitude * dom_force
                actual_v = np.abs(v_norm) + 1e-9
                
                z_trap = np.abs(expected_v) / actual_v
                trap_idx = np.argmax(z_trap)
                
                if z_trap[trap_idx] > 2.5:
                    trap_sym = symbols[trap_idx]
                    direction = 1 if expected_v[trap_idx] > 0 else -1
                    entry_price = sensorium[f"{trap_sym}_M1_price"].iloc[i]
                    
                    # FORWARD WALK: Thermodynamic Resolution Exit
                    j = i + 1
                    resolved = False
                    while j < len(sensorium):
                        fw_window = velocity_m15[j-window:j]
                        fw_mean = np.mean(fw_window, axis=0)
                        fw_std = np.std(fw_window, axis=0) + 1e-12
                        fw_norm = (fw_window[-1] - fw_mean) / fw_std
                        
                        fw_window_norm = (fw_window - fw_mean) / fw_std
                        fw_cov = np.cov(fw_window_norm.T)
                        fw_evals, fw_evecs = np.linalg.eigh(fw_cov)
                        fw_idx = np.argsort(fw_evals)[::-1]
                        fw_dom = fw_evecs[:, fw_idx[0]]
                        
                        fw_force = np.dot(fw_norm, fw_dom)
                        fw_exp = fw_force * fw_dom
                        fw_act = np.abs(fw_norm) + 1e-9
                        fw_z = np.abs(fw_exp) / fw_act
                        
                        # Exit when the trap's structural tension returns to baseline (< 1.5x)
                        if fw_z[trap_idx] <= 1.5: 
                            exit_price = sensorium[f"{trap_sym}_M1_price"].iloc[j]
                            disp = (exit_price - entry_price) * 100
                            
                            yield_val = disp if direction > 0 else -disp
                            
                            if yield_val > 0:
                                passes += 1
                                release_log.append(yield_val)
                            else:
                                failures += 1
                                loss_log.append(abs(yield_val))
                            
                            i = j # Advance the primary chronological clock to the exit state
                            resolved = True
                            break
                        j += 1
                    
                    if not resolved:
                        break # End of data reached without resolution
        
        prev_entropy = entropy
        i += 1

    win_rate = (passes / (passes + failures)) * 100 if (passes + failures) > 0 else 0
    avg_release = np.mean(release_log) if release_log else 0.0
    avg_loss = np.mean(loss_log) if loss_log else 0.0
    expectancy_ratio = avg_release / (avg_loss + 0.0001) if passes > 0 else 0.0
    ev = (win_rate/100 * avg_release) - ((1 - win_rate/100) * avg_loss)

    print("\n" + "="*80)
    print("AION DYNAMIC EXIT AUDIT: 6-MONTH STATISTICAL RESULTS")
    print("="*80)
    print(f"Total Traps Identified & Resolved: {passes + failures}")
    print(f"Mathematical Win Rate:             {win_rate:.2f}%")
    print(f"Average Kinetic Release (Yield):   {avg_release:.4f}%")
    print(f"Average Adverse Friction (Loss):   {avg_loss:.4f}%")
    print(f"System Expectancy Ratio:           {expectancy_ratio:.2f}x")
    print(f"Expected Value (EV):               {ev:+.5f}% per event")
    print("="*80)

if __name__ == "__main__":
    run_dynamic_exit_audit()
