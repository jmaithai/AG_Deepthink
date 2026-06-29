import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_propagation_audit():
    file_path = "archive/fractal_manifold_6M.parquet"
    print(f"[*] AION PROPAGATION AUDIT: Testing The 'Kinetic Catch-Up' Hypothesis...")
    
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
    
    close_cols = [c for c in df.columns if c.endswith('_M1_close')]
    symbols = [c.replace('_M1_close', '') for c in close_cols]
    
    sensorium = df.groupby('Event_Clock').last()
    sensorium['chronological_time'] = sensorium.index
    print(f"[+] Compressed into {len(sensorium):,} True Event States.")

    raw_prices = sensorium[close_cols].values
    velocity = np.diff(np.log(raw_prices + 1e-9), axis=0, prepend=np.log(raw_prices[0:1] + 1e-9))

    print("[*] Scanning Manifold for Global Currents and Frozen Dams...")
    
    window = 21
    rupture_states = []
    raw_df = df.copy()
    raw_df['chronological_time'] = raw_df.index
    
    for i in range(window, len(sensorium) - 5):
        v_window = velocity[i-window:i]
        
        mean_v = np.mean(v_window, axis=0)
        std_v = np.std(v_window, axis=0) + 1e-12
        
        v_window_norm = (v_window - mean_v) / std_v
        cov_matrix = np.cov(v_window_norm.T)
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eig_vals)[::-1]
        
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]
        
        # We need a strong global current (High power in dominant vector)
        total_energy = np.sum(eig_vals) + 1e-12
        p_energy = np.clip(eig_vals / total_energy, 1e-12, 1.0)
        
        if p_energy[0] > 0.40:
            dom_force = eig_vecs[:, 0]
            current_v_norm = (velocity[i] - mean_v) / std_v
            
            # The Physics of the Dam:
            # Expected Velocity (The Wave) = How hard the Dominant Vector is pushing the node
            # Actual Velocity = How fast the node is currently moving
            expected_v = dom_force * np.dot(current_v_norm, dom_force)
            
            # The Trap is the node where Expected is massive, but Actual is near zero (fighting the wave)
            tension_discrepancy = np.abs(expected_v) - np.abs(current_v_norm)
            
            # Find the node with the highest positive discrepancy (The Dam)
            if np.max(tension_discrepancy) > 1.0 and np.min(np.abs(current_v_norm)) < 0.5: 
                trap_idx = np.argmax(tension_discrepancy)
                trap_sym = symbols[trap_idx]
                
                # Direction of the trade: IN THE DIRECTION OF THE GLOBAL WAVE
                trade_direction = np.sign(expected_v[trap_idx])
                
                rupture_states.append({
                    'Time': sensorium['chronological_time'].iloc[i],
                    'Trap': trap_sym,
                    'Direction': trade_direction
                })

    ruptures_df = pd.DataFrame(rupture_states)
    if ruptures_df.empty:
        print("[-] No events found.")
        return
        
    ruptures_df['Time'] = pd.to_datetime(ruptures_df['Time'])
    ruptures_df['HourGroup'] = ruptures_df['Time'].dt.floor('H')
    distinct_ruptures = ruptures_df.drop_duplicates(subset=['HourGroup', 'Trap'])
    
    print(f"\n[+] Discovered {len(distinct_ruptures)} distinct 'Dam' formations over 6 months.")
    print("[*] Initiating Timeless Forward-Walk Falsification (100,000 tick energy injection)...")

    passes = 0
    failures = 0
    release_log = []
    loss_log = []
    
    for _, rupture in distinct_ruptures.iterrows():
        t0 = rupture['Time']
        trap_sym = rupture['Trap']
        direction = rupture['Direction']
        
        # --- THE TIMELESS FORWARD WALK ---
        post_rupture_mask = raw_df.index >= t0
        post_rupture_df = raw_df[post_rupture_mask].copy()
        
        if len(post_rupture_df) < 5: continue
            
        vol_cols_only = [c for c in post_rupture_df.columns if c.endswith('_M1_vol')]
        post_rupture_df['post_energy'] = post_rupture_df[vol_cols_only].sum(axis=1).cumsum()
        forward_df = post_rupture_df[post_rupture_df['post_energy'] <= 100000]
        
        if len(forward_df) < 5: continue
            
        p0 = forward_df[f"{trap_sym}_M1_close"].iloc[0]
        disp = np.log(forward_df[f"{trap_sym}_M1_close"] / p0) * 100
        
        max_up = disp.max()
        max_down = disp.min()
        
        # 3. FALSIFICATION LOGIC (Trading WITH the Global Flow)
        if direction > 0: # Global force demands an UP move
            release = max_up
            friction = abs(max_down)
        else: # Global force demands a DOWN move
            release = abs(max_down)
            friction = max_up

        # Did the dam break in the direction of the global flow?
        # Falsification logic: Release > 2x Friction
        if release > (friction * 2) and release > 0.01: 
            passes += 1
            release_log.append(release)
        else:
            failures += 1
            loss_log.append(friction)

    win_rate = (passes / (passes + failures)) * 100 if (passes + failures) > 0 else 0
    avg_release = np.mean(release_log) if release_log else 0.0
    avg_loss = np.mean(loss_log) if loss_log else 0.0
    expectancy_ratio = avg_release / (avg_loss + 0.0001) if passes > 0 else 0.0
    
    # Expected Value
    ev = (win_rate/100 * avg_release) - ((1 - win_rate/100) * avg_loss)

    print("\n" + "="*80)
    print("AION ENERGY FLOW (KINETIC CATCH-UP) AUDIT: RESULTS")
    print("="*80)
    print(f"Total Structural Dams Identified: {passes + failures}")
    print(f"Mathematical Win Rate (Forced Catch-Up): {win_rate:.2f}%")
    print(f"Average Kinetic Release:   {avg_release:.4f}% (Yield)")
    print(f"Average Adverse Friction:  {avg_loss:.4f}% (Drawdown)")
    print(f"System Expectancy Ratio:   {expectancy_ratio:.2f}x")
    print(f"Expected Value (EV):       {ev:+.5f}% per event")
    print("="*80)
    
    if ev > 0:
        print("[ VERDICT ] POSITIVE CONVEXITY DETECTED. THE MARKET IS A PROPAGATING WAVE.")
    else:
        print("[ VERDICT ] NEGATIVE EV. THE DAMS DO NOT BREAK IN THE DIRECTION OF THE WAVE.")

if __name__ == "__main__":
    run_propagation_audit()
