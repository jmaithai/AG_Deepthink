import pandas as pd
import numpy as np
import os
import warnings

warnings.simplefilter(action='ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_aion_law_engine():
    file_path = "archive/fractal_manifold_6M.parquet"
    print(f"[*] AION LAW ENGINE: Loading 6-Month 30-D Manifold from {file_path}")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Error: {e}")
        return

    print("[*] Rebuilding the Event Clock (25,000 physical volume units)...")
    vol_cols = [c for c in df.columns if c.endswith('_M1_vol')]
    df['network_ticks'] = df[vol_cols].sum(axis=1)
    df['cumulative_ticks'] = df['network_ticks'].cumsum()
    
    EVENT_THRESHOLD = 25000 
    df['Event_Clock'] = (df['cumulative_ticks'] // EVENT_THRESHOLD).astype(int)
    
    close_cols = [c for c in df.columns if c.endswith('_M1_close')]
    symbols = [c.replace('_M1_close', '') for c in close_cols]
    
    df['chronological_time'] = df.index
    sensorium = df.groupby('Event_Clock').last()
    
    print(f"[+] Compressed {len(df):,} chronological bars into {len(sensorium):,} Event States.")
    
    print("[*] Calculating Kinematics...")
    prices = np.log(sensorium[close_cols].values + 1e-9)
    velocity = np.diff(prices, axis=0, prepend=prices[0:1])
    
    print("[*] Scanning 6 Months for Wave-Function Collapses (Eigen-Decomposition)...")
    window = 21 
    rupture_states = []
    
    raw_df = df.copy()
    prev_degeneracy = 0
    
    for i in range(window, len(sensorium)):
        v_window = velocity[i-window:i]
        
        # Z-Score normalization per symbol
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
        
        if i > window:
            drop = S_degeneracy - prev_degeneracy
            if drop < -0.15: 
                vec = eigenvectors[:, 0]
                loadings = {symbols[j]: vec[j] for j in range(len(symbols))}
                sorted_loadings = sorted(loadings.items(), key=lambda item: abs(item[1]), reverse=True)
                
                epicenter = sorted_loadings[0][0]
                outlet = sorted_loadings[-1][0] 
                
                rupture_states.append({
                    'State_ID': i,
                    'Time': sensorium['chronological_time'].iloc[i],
                    'Epicenter': epicenter,
                    'Outlet': outlet
                })
                
        prev_degeneracy = S_degeneracy

    ruptures_df = pd.DataFrame(rupture_states)
    if ruptures_df.empty:
        print("[-] No ruptures found.")
        return
        
    ruptures_df['HourGroup'] = ruptures_df['Time'].dt.floor('H')
    distinct_ruptures = ruptures_df.drop_duplicates(subset=['HourGroup', 'Epicenter'])
    
    total_events = len(distinct_ruptures)
    print(f"\n[+] Discovered {total_events} distinct structural ruptures over 6 months.")
    print("[*] Initiating Thermodynamic Forward-Walk Falsification...")

    passes = 0
    failures = 0
    release_log = []
    loss_log = []
    
    for _, rupture in distinct_ruptures.iterrows():
        t0 = rupture['Time']
        epi = rupture['Epicenter']
        outlet = rupture['Outlet']
        
        # --- THE TIMELESS FORWARD WALK ---
        # Instead of 120 minutes, we wait for 100,000 physical network transactions
        post_rupture_mask = raw_df.index >= t0
        post_rupture_df = raw_df[post_rupture_mask].copy()
        
        if len(post_rupture_df) < 5:
            continue
            
        vol_cols_only = [c for c in post_rupture_df.columns if c.endswith('_M1_vol')]
        post_rupture_df['post_energy'] = post_rupture_df[vol_cols_only].sum(axis=1).cumsum()
        
        # Isolate the data block where cumulative energy is <= 100,000
        forward_df = post_rupture_df[post_rupture_df['post_energy'] <= 100000]
        
        if len(forward_df) < 5:
            continue
            
        p0_epi = forward_df[f"{epi}_M1_close"].iloc[0]
        p0_out = forward_df[f"{outlet}_M1_close"].iloc[0]
        
        epi_disp = np.log(forward_df[f"{epi}_M1_close"] / p0_epi) * 100
        out_disp = np.log(forward_df[f"{outlet}_M1_close"] / p0_out) * 100
        
        max_epi_ext = epi_disp.max() if abs(epi_disp.max()) > abs(epi_disp.min()) else epi_disp.min()
        max_out_ext = out_disp.max()
        max_out_drawdown = out_disp.min()
        
        if abs(max_out_drawdown) > abs(max_out_ext):
            release = abs(max_out_drawdown)
            friction = abs(max_out_ext)
        else:
            release = abs(max_out_ext)
            friction = abs(max_out_drawdown)

        # Falsification logic: Release > Illusion, Release > 2x Friction
        if release > abs(max_epi_ext) and release > (friction * 2):
            passes += 1
            release_log.append(release)
        else:
            failures += 1
            loss_log.append(friction)

    win_rate = (passes / (passes + failures)) * 100 if (passes + failures) > 0 else 0
    avg_release = np.mean(release_log) if release_log else 0.0
    avg_loss = np.mean(loss_log) if loss_log else 0.0
    
    expectancy_ratio = avg_release / (avg_loss + 0.0001) if passes > 0 else 0.0
    
    # Expected Value Formula: (Win% * Avg Win) - (Loss% * Avg Loss)
    ev = (win_rate/100 * avg_release) - ((1 - win_rate/100) * avg_loss)

    print("\n" + "="*80)
    print("AION LAW ENGINE: 6-MONTH TRUE STATISTICAL AUDIT (TIMELESS)")
    print("="*80)
    print(f"Total Manifold Ruptures Analyzed: {passes + failures}")
    print(f"Falsification Passes (Physics Held): {passes}")
    print(f"Falsification Failures (Physics Broke): {failures}")
    print(f"\n[ DOCTRINE METRICS ]")
    print(f"Mathematical Win Rate:     {win_rate:.2f}%")
    print(f"Average Kinetic Release:   {avg_release:.4f}% (Yield on Success)")
    print(f"Average Adverse Friction:  {avg_loss:.4f}% (Drawdown on Failure)")
    print(f"System Expectancy Ratio:   {expectancy_ratio:.2f}x (Release vs Loss)")
    print(f"Expected Value (EV):       {ev:+.5f}% per rupture")
    print("="*80)
    
    if ev > 0 and expectancy_ratio > 2.0:
        print("[ VERDICT ] POSITIVE CONVEXITY DETECTED. THE LAW IS PROVEN. READY FOR HARVEST.")
    else:
        print("[ VERDICT ] NEGATIVE EV. DOCTRINE REQUIRES CALIBRATION.")

if __name__ == "__main__":
    run_aion_law_engine()
