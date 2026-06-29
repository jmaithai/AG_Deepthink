import pandas as pd
import numpy as np
import os
import warnings
from sklearn.cluster import SpectralClustering
from scipy.linalg import expm

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def map_tectonic_plates():
    # Gauge Inversion Fixes
    file_path = "archive/fractal_manifold_6M.parquet"
    print("[*] AION OBSERVATORY: Initializing Tectonic Plate & Cascade Mapper...")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Data Error: {e}")
        return

    # 1. ERADICATE CHRONOLOGICAL TIME (Event Clock)
    vol_cols = [c for c in df.columns if c.endswith('_M1_vol')]
    df['network_ticks'] = df[vol_cols].sum(axis=1)
    df['cumulative_ticks'] = df['network_ticks'].cumsum()
    EVENT_THRESHOLD = int(df['network_ticks'].mean() * 1000) 
    df['Event_Clock'] = (df['cumulative_ticks'] // EVENT_THRESHOLD).astype(int)
    
    close_cols = [c for c in df.columns if c.endswith('_M15_price')]
    symbols = np.array([c.replace('_M15_price', '') for c in close_cols])
    
    sensorium = df.groupby('Event_Clock').last()
    
    # Values already log-transformed
    prices = sensorium[close_cols].values
    velocity = np.diff(prices, axis=0, prepend=prices[0:1])

    # 2. DIFFUSION MAPPING
    window = 100
    v_window = velocity[-window:]
    corr = np.corrcoef(v_window.T)
    
    adj = np.abs(corr) - np.eye(len(symbols))
    deg = np.diag(np.sum(adj, axis=1))
    laplacian = deg - adj
    
    # Diffusion Kernel: e^(-Lt)
    diffusion_kernel = expm(-laplacian * 1.0)
    
    # 3. SPECTRAL CLUSTERING (Self-Organizing the Manifold)
    n_clusters = 4 # We ask the math to find 4 macro-plates
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_init=10, random_state=42)
    
    # Ensure perfect symmetry for the affinity matrix
    affinity_matrix = (diffusion_kernel + diffusion_kernel.T) / 2.0
    np.fill_diagonal(affinity_matrix, 1.0)
    
    labels = sc.fit_predict(affinity_matrix)
    
    print("\n" + "="*85)
    print("AION OBSERVATORY: TECTONIC ENERGY PLATES (SPECTRAL CLUSTERS)")
    print("="*85)
    
    for cluster_id in range(n_clusters):
        cluster_nodes = symbols[labels == cluster_id]
        print(f"\n[ TECTONIC PLATE {cluster_id + 1} ]")
        print(f"Nodes: {', '.join(cluster_nodes)}")
        
    print("\n" + "="*85)
    print("AION FAULT LINE DETECTION & PROPAGATION CASCADES")
    print("="*85)
    
    # Identify the Kinetic Epicenter
    recent_v = np.abs(velocity[-1])
    epi_idx = np.argmax(recent_v)
    epicenter_sym = symbols[epi_idx]
    epi_cluster = labels[epi_idx] + 1
    
    print(f"[ KINETIC EPICENTER ]")
    print(f"Origin Node: [{epicenter_sym}] (Tectonic Plate {epi_cluster})")
    print(f"State: Experiencing highest localized kinetic velocity.")
    
    # Strongest Inter-Plate Thread (The Fault Line)
    np.fill_diagonal(affinity_matrix, 0)
    max_leak = 0
    leak_source = ""
    leak_target = ""
    
    # Check all pairs. If they are in different tectonic plates, measure the thread.
    for i in range(len(symbols)):
        for j in range(len(symbols)):
            if labels[i] != labels[j]:
                if affinity_matrix[i, j] > max_leak:
                    max_leak = affinity_matrix[i, j]
                    leak_source = symbols[i]
                    leak_target = symbols[j]
                    
    print(f"\n[ PRIMARY FAULT LINE (INTER-PLATE LEAK) ]")
    print(f"[{leak_source}] is bleeding {max_leak:.4f} units of kinetic energy into [{leak_target}] across tectonic boundaries.")
    
    # Propagation Cascade
    diffusion_from_epi = diffusion_kernel[epi_idx]
    # Get top 5 receivers, skipping the 0th index if it happens to be the node itself
    top_receivers_idx = np.argsort(diffusion_from_epi)[::-1]
    top_receivers_idx = [x for x in top_receivers_idx if x != epi_idx][:5]
    
    print(f"\n[ THE SILK THREADS (Propagation Cascade) ]")
    print(f"When energy injects at [{epicenter_sym}], the topology dictates it MUST flow to:")
    for rank, r_idx in enumerate(top_receivers_idx, 1):
        target_sym = symbols[r_idx]
        target_cluster = labels[r_idx] + 1
        intensity = diffusion_from_epi[r_idx]
        print(f"  {rank}. [{target_sym:<8}] (Plate {target_cluster}) | Diffusion Intensity: {intensity:.5f}")

    print("="*85)

if __name__ == "__main__":
    map_tectonic_plates()
