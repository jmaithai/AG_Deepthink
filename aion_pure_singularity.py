import pandas as pd
import numpy as np
import os
import warnings
from scipy.stats import gaussian_kde

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_pure_singularity_engine():
    file_path = "archive/fractal_manifold_6M.parquet"
    print("[*] AION PURE SINGULARITY: Initializing Parameter-less Fluid Dynamics...")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Critical Error: {e}")
        return

    vol_cols = [c for c in df.columns if c.endswith('_M1_vol')]
    close_cols = [c for c in df.columns if c.endswith('_M1_price')]
    symbols = np.array([c.replace('_M1_price', '') for c in close_cols])

    # 1. DYNAMIC EVENT CLOCK (Variance Thresholding)
    print("[*] Calculating Dynamic Systemic Variance Clock...")
    network_vol = df[vol_cols].sum(axis=1).values
    
    global_vol_mean = np.mean(network_vol)
    global_vol_std = np.std(network_vol) + 1e-9
    
    # Threshold = Mean + 5 Sigma. Purely dynamic to the physical variance of the asset class.
    EVENT_THRESHOLD = int(global_vol_mean + (5.0 * global_vol_std))
    
    df['cumulative_ticks'] = np.cumsum(network_vol)
    df['Event_Clock'] = (df['cumulative_ticks'] // EVENT_THRESHOLD).astype(int)
    
    sensorium = df.groupby('Event_Clock').last()
    if 'time' in df.columns:
        sensorium['chronological_time'] = df.groupby('Event_Clock')['time'].last()
    else:
        sensorium['chronological_time'] = df.groupby('Event_Clock').apply(lambda x: x.index[-1]).values
    vol_df = df.groupby('Event_Clock')[vol_cols].sum()
    
    print(f"[*] Dynamic Clock Compressed {len(df):,} ticks into {len(sensorium):,} Phase States.")

    prices = sensorium[close_cols].values
    velocity = np.diff(prices, axis=0, prepend=prices[0:1])
    
    # 4. DYNAMIC LOOKBACK WINDOW (Autocorrelation Wavelength)
    print("[*] Calculating Dynamic Manifold Wavelength (FFT / Autocorrelation)...")
    total_kinetic = np.sum(np.abs(velocity), axis=1)
    
    def estimate_wavelength(signal, max_lag=200):
        if len(signal) < max_lag: return max(10, len(signal)//2)
        signal = signal - np.mean(signal)
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
        if len(zero_crossings) == 0: return 50
        first_zero = zero_crossings[0]
        if first_zero >= max_lag: return max_lag
        peak_idx = first_zero + np.argmax(autocorr[first_zero:max_lag])
        return max(15, peak_idx)

    dynamic_window = estimate_wavelength(total_kinetic)
    print(f"[*] Manifold Wavelength Locked: {dynamic_window} States.")
    
    singularities = []
    
    if len(sensorium) < dynamic_window + 50:
        print("[-] Fatal: Resolution too low for dynamic memory mapping.")
        return
        
    print("[*] Hunting 3-Sigma Relative-Entropy Singularities...")
    
    for i in range(dynamic_window, len(sensorium) - 50):
        v_window = velocity[i-dynamic_window:i]
        
        std_v = np.std(v_window, axis=0) + 1e-12
        v_norm = v_window / std_v
        
        corr = np.corrcoef(v_norm.T)
        np.fill_diagonal(corr, 0)
        adj = np.clip(corr, 0, None) 
        
        deg = np.diag(np.sum(adj, axis=1))
        laplacian = deg - adj
        
        current_v = velocity[i] / std_v
        network_force = -np.dot(laplacian, current_v)
        
        alphas = np.zeros(len(symbols))
        valid_traps = []
        
        # Calculate Alpha for ALL nodes at this exact timestamp to find the Manifold Baseline
        for idx in range(len(symbols)):
            hist_p = prices[i-dynamic_window:i, idx]
            p_min, p_max = np.min(hist_p), np.max(hist_p)
            if p_max == p_min: continue
                
            # 2. DYNAMIC TOPOGRAPHY (Gaussian KDE with Silverman's Rule)
            try:
                kde = gaussian_kde(hist_p, bw_method='silverman')
            except np.linalg.LinAlgError:
                continue # Singular matrix (no variance)
                
            curr_p = prices[i, idx]
            local_density = kde(curr_p)[0]
            
            eval_points = np.linspace(p_min, p_max, 50)
            avg_density = np.mean(kde(eval_points)) + 1e-6
            
            push_up = network_force[idx] > 0
            p_range = p_max - p_min
            
            if push_up:
                void_eval = np.linspace(curr_p, curr_p + (p_range * 0.1), 10)
            else:
                void_eval = np.linspace(curr_p - (p_range * 0.1), curr_p, 10)
                
            void_density = np.mean(kde(void_eval))
            
            norm_force = np.abs(network_force[idx])
            alpha_val = (norm_force * (local_density / avg_density)) / ((void_density / avg_density) + 1e-6)
            alphas[idx] = alpha_val
            
            # Opposing actual velocity ensures it's a real trap
            if np.sign(network_force[idx]) != np.sign(current_v[idx]) or abs(current_v[idx]) < 0.1:
                valid_traps.append(idx)
        
        # 3. DYNAMIC SINGULARITY TRIGGER (Relative Entropy Degeneracy > 3 Sigma)
        mean_alpha = np.mean(alphas)
        std_alpha = np.std(alphas) + 1e-6
        
        for trap_idx in valid_traps:
            node_alpha = alphas[trap_idx]
            
            if node_alpha > mean_alpha + (3.0 * std_alpha) and node_alpha > 5.0:
                trap_sym = symbols[trap_idx]
                push_up = network_force[trap_idx] > 0
                direction = "UPWARD" if push_up else "DOWNWARD"
                
                p0 = prices[i, trap_idx]
                p_future = prices[i+1:i+21, trap_idx]
                
                if push_up:
                    release = abs(np.max(p_future) - p0) * 100
                    friction = abs(np.min(p_future) - p0) * 100
                else:
                    release = abs(np.min(p_future) - p0) * 100
                    friction = abs(np.max(p_future) - p0) * 100
                    
                singularities.append({
                    'Time': sensorium['chronological_time'].iloc[i],
                    'Symbol': trap_sym,
                    'Direction': direction,
                    'Alpha': node_alpha,
                    'Sigma_Deviation': (node_alpha - mean_alpha) / std_alpha,
                    'Release': release,
                    'Friction': friction
                })

    print("[*] Processing Rupture Telemetry...")
    if not singularities:
        print("[-] No singularities detected. Physics remain in bounds.")
        return
        
    df_sing = pd.DataFrame(singularities)
    
    passes = df_sing[(df_sing['Release'] > df_sing['Friction']) & (df_sing['Release'] > 0.05)]
    win_rate = (len(passes) / len(df_sing)) * 100 if len(df_sing) > 0 else 0
    
    print("\n" + "="*85)
    print("AION DEEP THINK: PARAMETER-LESS FLUID DYNAMICS (PURE SINGULARITY)")
    print("="*85)
    print(f"Dynamic Wavelength (Lookback):            {dynamic_window} States")
    print(f"Total 3-Sigma Singularities Detected:     {len(df_sing)}")
    print(f"Mathematical Win Rate (Release>Friction): {win_rate:.2f}%")
    print(f"Average Kinetic Yield:                    {df_sing['Release'].mean():.4f}%")
    print("="*85)
    
    df_sing['Hour'] = df_sing['Time'].dt.floor('H')
    top_5 = df_sing.sort_values(by='Sigma_Deviation', ascending=False).drop_duplicates(subset=['Symbol', 'Hour']).head(5)
    
    for rank, (_, row) in enumerate(top_5.iterrows(), 1):
        ts = row['Time'].strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[ 3\u03C3 RUPTURE {rank} | T=0: {ts} ]")
        print(f"WHAT is happening:  [{row['Symbol']}] reached {row['Sigma_Deviation']:.1f}\u03C3 Outlier State (\u03B1 = {row['Alpha']:,.1f})")
        print(f"WHY is it doing it: Immense force ({row['Direction']}) against dynamic KDE density wall.")
        print(f"HOW do we gain:     The wall exhausted into a true Phase-Space Vacuum.")
        print(f"PHYSICAL OUTCOME:   Node successfully teleported {row['Release']:.4f}% into the vacuum.")
        print(f"Adverse Friction:   {row['Friction']:.4f}%")
    print("="*85)

if __name__ == "__main__":
    run_pure_singularity_engine()
