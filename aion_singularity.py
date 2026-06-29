import pandas as pd
import numpy as np
import os
import warnings
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_singularity_engine():
    file_path = "archive/fractal_manifold_6M.parquet"
    print("[*] AION DEEP THINK: Initializing Navier-Stokes Financial Fluid Dynamics...")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Critical Error: {e}")
        return

    # Gauge Inversion Fixes
    vol_cols = [c for c in df.columns if c.endswith('_M1_vol')]
    close_cols = [c for c in df.columns if c.endswith('_M1_price')]
    symbols = np.array([c.replace('_M1_price', '') for c in close_cols])

    # 1. THE THERMODYNAMIC CLOCK
    print("[*] Eradicating Chronological Time...")
    df['network_ticks'] = df[vol_cols].sum(axis=1)
    df['cumulative_ticks'] = df['network_ticks'].cumsum()
    
    # Fix the Threshold to retain high resolution of thermodynamic states (~20,000 states)
    # The Architect originally used 5000, which mathematically over-compressed the dataset (dividing by a massive number)
    EVENT_THRESHOLD = int(df['network_ticks'].mean() * 100) 
    df['Event_Clock'] = (df['cumulative_ticks'] // EVENT_THRESHOLD).astype(int)
    
    sensorium = df.groupby('Event_Clock').last()
    
    if 'time' in df.columns:
        sensorium['chronological_time'] = df.groupby('Event_Clock')['time'].last()
    else:
        sensorium['chronological_time'] = df.groupby('Event_Clock').apply(lambda x: x.index[-1]).values
        
    vol_df = df.groupby('Event_Clock')[vol_cols].sum()
    
    print(f"[*] Compressed {len(df):,} raw ticks into {len(sensorium):,} True Phase States.")

    prices = sensorium[close_cols].values
    # Gauge Inversion: prices are already log-transformed
    velocity = np.diff(prices, axis=0, prepend=prices[0:1])
    volumes = vol_df.values

    print("[*] Calculating Unified Field Equation (Network Force vs Phase-Space Topography)...")
    
    window = 100 
    singularities = []
    
    if len(sensorium) < window + 50:
        print("[-] Fatal: Resolution too low for topographical mapping. Need smaller EVENT_THRESHOLD.")
        return
    
    for i in range(window, len(sensorium) - 50):
        # --- A. NETWORK DIFFUSION (The Web's Pull) ---
        v_window = velocity[i-window:i]
        
        # Normalize
        std_v = np.std(v_window, axis=0) + 1e-12
        v_norm = v_window / std_v
        
        corr = np.corrcoef(v_norm.T)
        np.fill_diagonal(corr, 0)
        adj = np.clip(corr, 0, None) 
        
        deg = np.diag(np.sum(adj, axis=1))
        laplacian = deg - adj
        
        current_v = velocity[i] / std_v
        
        # The exact mathematical force the entire network is exerting on each node
        network_force = -np.dot(laplacian, current_v)
        
        # We map the topography ONLY for the node under the absolute highest network stress
        trap_idx = np.argmax(np.abs(network_force))
        trap_sym = symbols[trap_idx]
        
        # Check if the node is actually fighting the force (The Trap)
        if np.sign(network_force[trap_idx]) == np.sign(current_v[trap_idx]) and abs(current_v[trap_idx]) > 0.1:
            continue # It is flowing freely with the wave. No trap exists.
            
        # --- B. LOCAL DENSITY (The Concrete Wall & The Vacuum) ---
        hist_p = prices[i-window:i, trap_idx]
        hist_v = volumes[i-window:i, trap_idx]
        
        p_min, p_max = np.min(hist_p), np.max(hist_p)
        if p_max == p_min: continue
            
        # Construct the Phase-Space Volume Profile
        bins = np.linspace(p_min, p_max, 50)
        digitized = np.clip(np.digitize(hist_p, bins) - 1, 0, 49)
        
        vol_profile = np.zeros(50)
        for k in range(len(hist_p)):
            vol_profile[digitized[k]] += hist_v[k]
                
        # Smooth the profile to simulate Thermodynamic Heat Diffusion across price levels
        smoothed_profile = gaussian_filter1d(vol_profile, sigma=1.5)
        
        curr_p = prices[i, trap_idx]
        curr_bin = np.clip(np.digitize([curr_p], bins)[0] - 1, 0, 49)
        
        if not (5 < curr_bin < 45): continue
        
        local_density = smoothed_profile[curr_bin]
        
        # Force direction
        push_up = network_force[trap_idx] > 0
        direction = "UPWARD" if push_up else "DOWNWARD"
        
        if push_up:
            void_density = np.mean(smoothed_profile[curr_bin+1:curr_bin+6])
        else:
            void_density = np.mean(smoothed_profile[curr_bin-5:curr_bin])
            
        # --- C. THE SINGULARITY INDEX (\u03B1) ---
        norm_force = np.abs(network_force[trap_idx])
        avg_density = np.mean(smoothed_profile) + 1e-6
        
        # High Force * High Local Compression / Empty Void Ahead
        alpha = (norm_force * (local_density / avg_density)) / ((void_density / avg_density) + 1e-6)
        
        # Trigger Condition: Alpha approaches Singularity
        if alpha > 25.0 and local_density > avg_density: 
            
            p0 = prices[i, trap_idx]
            p_future = prices[i+1:i+21, trap_idx]
            
            # Gauge Inversion Fix: Prices are already log-transformed
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
                'Alpha': alpha,
                'Network_Force': norm_force,
                'Release': release,
                'Friction': friction
            })

    print("[*] Processing Rupture Telemetry...")
    if not singularities:
        print("[-] No singularities detected. Physics remain in bounds.")
        return
        
    df_sing = pd.DataFrame(singularities)
    
    # Falsification: Teleportation is real if release > friction
    passes = df_sing[(df_sing['Release'] > df_sing['Friction']) & (df_sing['Release'] > 0.05)]
    win_rate = (len(passes) / len(df_sing)) * 100 if len(df_sing) > 0 else 0
    
    print("\n" + "="*85)
    print("AION DEEP THINK: UNIFIED FIELD FLUID DYNAMICS (SINGULARITY ENGINE)")
    print("="*85)
    print(f"Total Structural Singularities Detected: {len(df_sing)}")
    print(f"Mathematical Win Rate (Vacuum Teleportation): {win_rate:.2f}%")
    print(f"Average Kinetic Yield: {df_sing['Release'].mean():.4f}%")
    print("="*85)
    
    df_sing['Hour'] = df_sing['Time'].dt.floor('H')
    top_5 = df_sing.sort_values(by='Alpha', ascending=False).drop_duplicates(subset=['Symbol', 'Hour']).head(5)
    
    for rank, (_, row) in enumerate(top_5.iterrows(), 1):
        ts = row['Time'].strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[ SINGULARITY {rank} | T=0: {ts} ]")
        print(f"WHAT is happening:  [{row['Symbol']}] reached Critical Singularity Index (\u03B1 = {row['Alpha']:,.1f})")
        print(f"WHY is it doing it: The Global Graph Laplacian applied immense force ({row['Direction']}) against a local concrete wall.")
        print(f"HOW do we gain:     The local volume wall exhausted. A liquidity vacuum existed directly behind it.")
        print(f"PHYSICAL OUTCOME:   Node successfully teleported {row['Release']:.4f}% into the vacuum.")
        print(f"Adverse Friction:   {row['Friction']:.4f}%")
    print("="*85)

if __name__ == "__main__":
    run_singularity_engine()
