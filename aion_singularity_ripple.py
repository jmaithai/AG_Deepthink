import pandas as pd
import numpy as np
import os
import warnings
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_ripple_engine():
    file_path = "archive/fractal_manifold_6M.parquet"
    print("[*] AION RIPPLE ENGINE: Initializing Multi-Dimensional Shockwave Analysis...")
    
    try:
        df = pd.read_parquet(file_path)
        # Filter to last month of data
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            threshold_date = df['time'].max() - pd.Timedelta(days=30)
            df = df[df['time'] >= threshold_date].copy()
        else:
            df.index = pd.to_datetime(df.index)
            threshold_date = df.index.max() - pd.Timedelta(days=30)
            df = df[df.index >= threshold_date].copy()
        print(f"[*] Filtered dataset to last month. Total ticks: {len(df):,}")
    except Exception as e:
        print(f"[!] Critical Error: {e}")
        return

    vol_cols = [c for c in df.columns if c.endswith('_M1_vol')]
    close_cols = [c for c in df.columns if c.endswith('_M1_price')]
    symbols = np.array([c.replace('_M1_price', '') for c in close_cols])

    print("[*] Calculating Dynamic Systemic Variance Clock...")
    network_vol = df[vol_cols].sum(axis=1).values
    
    global_vol_mean = np.mean(network_vol)
    global_vol_std = np.std(network_vol) + 1e-9
    
    # Increase multiplier to compress to ~1,000 states for rapid prototype execution
    EVENT_THRESHOLD = int(global_vol_mean + (200.0 * global_vol_std))
    
    df['cumulative_ticks'] = np.cumsum(network_vol)
    df['Event_Clock'] = (df['cumulative_ticks'] // EVENT_THRESHOLD).astype(int)
    
    sensorium = df.groupby('Event_Clock').last()
    if 'time' in df.columns:
        sensorium['chronological_time'] = df.groupby('Event_Clock')['time'].last()
    else:
        sensorium['chronological_time'] = df.groupby('Event_Clock').apply(lambda x: x.index[-1]).values
        
    print(f"[*] Dynamic Clock Compressed {len(df):,} ticks into {len(sensorium):,} Phase States.")

    prices = sensorium[close_cols].values
    velocity = np.diff(prices, axis=0, prepend=prices[0:1])
    
    dynamic_window = 100 # Fixed for rapid testing
    singularities = []
    
    if len(sensorium) < dynamic_window + 50:
        print("[-] Fatal: Resolution too low for dynamic memory mapping.")
        return
        
    print("[*] Hunting 3-Sigma Singularities & Mapping Secondary Ripples...")
    
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
        kdes = {}
        
        for idx in range(len(symbols)):
            hist_p = prices[i-dynamic_window:i, idx]
            p_min, p_max = np.min(hist_p), np.max(hist_p)
            if p_max == p_min: continue
                
            try:
                kde = gaussian_kde(hist_p, bw_method='silverman')
                kdes[idx] = kde
            except np.linalg.LinAlgError:
                continue
                
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
            
            if np.sign(network_force[idx]) != np.sign(current_v[idx]) or abs(current_v[idx]) < 0.1:
                valid_traps.append(idx)
        
        mean_alpha = np.mean(alphas)
        std_alpha = np.std(alphas) + 1e-6
        
        for trap_idx in valid_traps:
            node_alpha = alphas[trap_idx]
            
            if node_alpha > mean_alpha + (3.0 * std_alpha) and node_alpha > 5.0:
                trap_sym = symbols[trap_idx]
                push_up = network_force[trap_idx] > 0
                direction = "UPWARD" if push_up else "DOWNWARD"
                
                # --- THERMODYNAMIC RISK SIZING ---
                kde = kdes[trap_idx]
                hist_p = prices[i-dynamic_window:i, trap_idx]
                eval_p = np.linspace(np.min(hist_p), np.max(hist_p), 100)
                density = kde(eval_p)
                
                # Find KDE peaks to identify the center of the structural wall
                peaks, _ = find_peaks(density)
                if len(peaks) > 0:
                    main_wall_idx = peaks[np.argmax(density[peaks])]
                    wall_price = eval_p[main_wall_idx]
                else:
                    wall_price = prices[i, trap_idx]
                
                # Dynamic Stop Loss is placed on the opposite side of the wall
                p_range = np.max(hist_p) - np.min(hist_p)
                if push_up:
                    stop_loss_distance = (prices[i, trap_idx] - (wall_price - p_range * 0.2))
                else:
                    stop_loss_distance = ((wall_price + p_range * 0.2) - prices[i, trap_idx])
                stop_loss_distance = max(stop_loss_distance, 0.001) # Avoid zero distance
                
                sigma_dev = (node_alpha - mean_alpha) / std_alpha
                risk_allocation = min(5.0, max(1.0, 1.0 + (sigma_dev - 3.0) * 2.0)) # 1% for 3 sigma, up to 5% for 5 sigma
                
                # --- THE RIPPLE: MULTI-DIMENSIONAL TRAVERSAL ---
                corr_row = adj[trap_idx]
                # Find the highest correlated node (that isn't the node itself)
                corr_row[trap_idx] = -1 
                secondary_idx = np.argmax(corr_row)
                secondary_sym = symbols[secondary_idx]
                secondary_corr = corr_row[secondary_idx]
                
                singularities.append({
                    'Time': sensorium['chronological_time'].iloc[i],
                    'Primary_Symbol': trap_sym,
                    'Direction': direction,
                    'Sigma': sigma_dev,
                    'Risk_Percent': risk_allocation,
                    'Stop_Distance': stop_loss_distance,
                    'Ripple_Target': secondary_sym,
                    'Ripple_Correlation': secondary_corr
                })

    print("[*] Processing Phase 3 Telemetry...")
    if not singularities:
        print("[-] No singularities detected.")
        return
        
    df_sing = pd.DataFrame(singularities)
    
    print("\n" + "="*85)
    print("AION DEEP THINK: RIPPLE ENGINE (THERMODYNAMIC RISK & SHOCKWAVE TRAVERSAL)")
    print("="*85)
    print(f"Total Singularities Detected: {len(df_sing)}")
    print("="*85)
    
    df_sing['Hour'] = df_sing['Time'].dt.floor('H')
    top_events = df_sing.sort_values(by='Sigma', ascending=False).drop_duplicates(subset=['Primary_Symbol', 'Hour']).head(5)
    
    for rank, (_, row) in enumerate(top_events.iterrows(), 1):
        ts = row['Time'].strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[ EVENT {rank} | T=0: {ts} ]")
        print(f"PRIMARY RUPTURE: [{row['Primary_Symbol']}] {row['Sigma']:.1f}\u03C3 {row['Direction']} into Vacuum.")
        print(f"THE SHIELD:      Risk sizing calculated at {row['Risk_Percent']:.2f}% of capital.")
        print(f"                 Dynamic Structural Stop placed {row['Stop_Distance']*100:.2f}% behind the KDE Wall.")
        print(f"THE RIPPLE:      Shockwave propagating to [{row['Ripple_Target']}].")
        print(f"                 Adjacency Correlation Weight: {row['Ripple_Correlation']:.3f}")
    print("="*85)

if __name__ == "__main__":
    run_ripple_engine()
