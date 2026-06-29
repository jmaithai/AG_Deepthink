import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def map_propagation():
    # Gauge Inversion Fix: Path matches actual parquet
    file_path = "archive/fractal_manifold_6M.parquet"
    print("[*] AION OBSERVATORY: Initializing Field Propagation Mapper...")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Data Error: {e}")
        return

    # 1. ERADICATE CHRONOLOGICAL TIME (Event Clock)
    # Gauge Inversion Fix: suffixes
    vol_cols = [c for c in df.columns if c.endswith('_M1_vol')]
    df['network_ticks'] = df[vol_cols].sum(axis=1)
    df['cumulative_ticks'] = df['network_ticks'].cumsum()
    EVENT_THRESHOLD = int(df['network_ticks'].mean() * 1000) 
    df['Event_Clock'] = (df['cumulative_ticks'] // EVENT_THRESHOLD).astype(int)
    
    close_cols = [c for c in df.columns if c.endswith('_M15_price')]
    symbols = [c.replace('_M15_price', '') for c in close_cols]
    
    sensorium = df.groupby('Event_Clock').last()
    
    # Gauge Inversion Fix: Values already log-transformed
    prices = sensorium[close_cols].values
    velocity = np.diff(prices, axis=0, prepend=prices[0:1])

    # 2. DIFFUSION MAPPING (The Silk Threads)
    # We use the Graph Laplacian to map the connectivity structure
    window = 50
    v_window = velocity[-window:]
    corr = np.corrcoef(v_window.T)
    
    # Laplacian L = D - A
    adj = np.abs(corr) - np.eye(len(symbols))
    deg = np.diag(np.sum(adj, axis=1))
    laplacian = deg - adj
    
    # Diffusion Kernel: e^(-Lt)
    # This matrix maps how energy propagates from any node i to any node j
    t = 1.0 
    from scipy.linalg import expm
    diffusion_kernel = expm(-laplacian * t)
    
    print("\n" + "="*85)
    print("AION OBSERVATORY: FIELD PROPAGATION MAP")
    print("="*85)
    
    # Identify the strongest thread
    np.fill_diagonal(diffusion_kernel, 0)
    idx_flat = np.argmax(diffusion_kernel)
    i, j = np.unravel_index(idx_flat, diffusion_kernel.shape)
    
    print(f"[+] Strongest Silk Thread detected: [{symbols[i]}] -> [{symbols[j]}]")
    print(f"[+] Diffusion Intensity: {diffusion_kernel[i, j]:.4f}")
    
    # Calculate the structural pull on the target node (j)
    mean_v = np.mean(v_window, axis=0)
    std_v = np.std(v_window, axis=0) + 1e-12
    v_norm = (v_window[-1] - mean_v) / std_v
    
    v_window_norm = (v_window - mean_v) / std_v
    cov_matrix = np.cov(v_window_norm.T)
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
    idx_sort = np.argsort(eig_vals)[::-1]
    dom_force = eig_vecs[:, idx_sort[0]]
    
    force_magnitude = np.dot(v_norm, dom_force)
    expected_v = force_magnitude * dom_force
    
    direction = "LONG (BUY)" if expected_v[j] > 0 else "SHORT (SELL)"
    
    print("\n[ ACTION PHYSICS ARBITER ]")
    print(f"Target Node: [{symbols[j]}]")
    print(f"Structural Pull: {expected_v[j]:+.2f} (Expected Velocity)")
    print(f"Actual Kinetic Velocity: {v_norm[j]:+.2f}")
    print(f"Verdict: IF local volume is fighting this pull, prepare to execute {direction} on [{symbols[j]}].")
    
    print("\n[ FIELD DYNAMICS ]")
    print("The Observatory is now mapping the total network tension.")
    print("Energy currently propagates through the manifold as a function of its self-organized topology.")
    print("We are observing the market's hidden connective tissue.")
    print("="*85)

if __name__ == "__main__":
    map_propagation()
