import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_spiderweb_mapper():
    file_path = "archive/fractal_manifold_6M.parquet"
    print("[*] AION SPIDERWEB CORE: Mapping the Silk Threads of the Manifold...")
    
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

    prices = np.log(sensorium[close_cols].values + 1e-9)

    print("[*] Constructing Graph Laplacian & Calculating Network Tension...")
    
    window = 100 # Long window to map the true structural "threads" of the web
    passes = 0
    failures = 0
    
    for i in range(window, len(sensorium) - 10):
        # 1. Historical Structure (The Web)
        prices_hist = prices[i-window:i]
        
        # Adjacency Matrix (A) based on velocity correlation (The Silk Threads)
        vel_hist = np.diff(prices_hist, axis=0)
        corr_matrix = np.corrcoef(vel_hist.T)
        np.fill_diagonal(corr_matrix, 0) # No self-loops
        
        # We only care about strong structural threads
        A = np.where(np.abs(corr_matrix) > 0.5, corr_matrix, 0)
        
        # Degree Matrix (D)
        D = np.diag(np.sum(np.abs(A), axis=1))
        
        # Graph Laplacian (L = D - A)
        # L dictates how energy diffuses through the network
        L = D - A 
        
        # 2. The Recent Shock (The Fly hitting the web)
        # Velocity over the last 3 Event States (Kinetic Injection)
        recent_velocity = prices[i] - prices[i-3]
        
        # 3. The Pull of the Web
        # The force exerted on each node by its neighbors based on the Laplacian
        # Formula: Pull = -L * Velocity
        network_pull = -np.dot(L, recent_velocity)
        
        # 4. Tension Discrepancy (The Lagging Receiver)
        # We look for a node the network is pulling violently, but is physically frozen.
        
        expected_mag = np.abs(network_pull)
        actual_mag = np.abs(recent_velocity)
        
        # Trap Score: High Pull / Low Actual Movement
        trap_score = expected_mag / (actual_mag + 1e-6)
        
        # Trigger Condition: The network demands a massive move, but the node is stuck.
        if np.max(trap_score) > 10.0 and np.max(expected_mag) > np.percentile(expected_mag, 90):
            target_idx = np.argmax(trap_score)
            target_sym = symbols[target_idx]
            
            # The Insight: Which nodes are pulling it?
            pulling_nodes = np.argsort(np.abs(A[target_idx]))[::-1][:3]
            pullers = [symbols[idx] for idx in pulling_nodes]
            
            # The Action: Trade IN THE DIRECTION of the network's pull. Catch the snap.
            direction_of_pull = np.sign(network_pull[target_idx])
            
            # 5. Forward Walk Falsification
            # Does the node snap in the direction the web is pulling it over the next 10 states?
            future_displacement = prices[i+10, target_idx] - prices[i, target_idx]
            
            # Did it catch up to the network?
            if np.sign(future_displacement) == direction_of_pull and abs(future_displacement) > actual_mag[target_idx]:
                passes += 1
            else:
                failures += 1

    win_rate = (passes / (passes + failures)) * 100 if (passes + failures) > 0 else 0
    print("\n" + "="*80)
    print("AION SPIDERWEB CORE (NETWORK PROPAGATION AUDIT)")
    print("="*80)
    print(f"Total Network Lag Traps Identified: {passes + failures}")
    print(f"Kinetic Catch-Up Win Rate: {win_rate:.2f}%")
    print("="*80)
    if win_rate > 55:
        print("[ VERDICT ] THE WEB IS MAPPED. PROPAGATION ARBITRAGE CONFIRMED.")
    else:
        print("[ VERDICT ] NOISE DETECTED.")

if __name__ == "__main__":
    run_spiderweb_mapper()
