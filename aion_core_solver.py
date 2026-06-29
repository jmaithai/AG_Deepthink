import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def solve_the_problem():
    file_path = "archive/fractal_manifold_6M.parquet"
    print("[*] AION CORE SOLVER: Ingesting Omni-Dimensional Energy Field...")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Error: {e}")
        return

    # 1. ERADICATING TIME (The Event Clock)
    print("[*] Converting Chronological Time to Pure Thermodynamic States...")
    vol_cols = [c for c in df.columns if c.endswith('_M1_vol')]
    df['network_ticks'] = df[vol_cols].sum(axis=1)
    df['cumulative_ticks'] = df['network_ticks'].cumsum()
    
    # Event clock ticks scale naturally with network density
    EVENT_THRESHOLD = int(df['network_ticks'].mean() * 1000) 
    df['Event_Clock'] = (df['cumulative_ticks'] // EVENT_THRESHOLD).astype(int)
    
    close_cols = [c for c in df.columns if c.endswith('_M15_price')]
    symbols = [c.replace('_M15_price', '') for c in close_cols]
    
    df['chronological_time'] = df.index
    sensorium = df.groupby('Event_Clock').last()
    
    print(f"[*] Compressed into {len(sensorium):,} Event States.")

    # 2. STATE VECTORS
    raw_prices = sensorium[close_cols].values
    
    # Spatial Normalization (Z-Score of prices across the manifold)
    prices = (raw_prices - np.mean(raw_prices, axis=0)) / (np.std(raw_prices, axis=0) + 1e-9)
    
    volumes = sensorium[vol_cols].values
    mass = (volumes - np.mean(volumes, axis=0)) / (np.std(volumes, axis=0) + 1e-9)
    mass = np.clip(mass + 1.0, 0.1, None) # Mass must be strictly positive
    
    velocity = np.diff(prices, axis=0, prepend=prices[0:1])
    
    # 3. KINETIC ENERGY (T)
    T = 0.5 * mass * (velocity**2)
    
    print("[*] Mapping the Topology and Hamiltonian Energies...")
    
    window = 50 # Rolling window for topological memory
    V = np.zeros_like(T)
    
    # 4. POTENTIAL ENERGY (V) & TOPOLOGY
    for i in range(window, len(sensorium)):
        v_window = velocity[i-window:i]
        
        # Calculate dynamic correlations (The Silk Threads)
        corr_matrix = np.corrcoef(v_window.T)
        np.fill_diagonal(corr_matrix, 0)
        
        # We only care about strong positive correlations for structural tethering
        adj_matrix = np.clip(corr_matrix, 0, None)
        
        # Network-Implied Equilibrium for each node
        # The expected position of node j based on the nodes it is tethered to
        weights_sum = np.sum(adj_matrix, axis=1) + 1e-9
        network_equilibrium = np.dot(adj_matrix, prices[i]) / weights_sum
        
        # Potential Energy: Mass * Squared distance from Network Equilibrium
        displacement = prices[i] - network_equilibrium
        V[i] = 0.5 * mass[i] * (displacement**2)

    # 5. TOTAL ENERGY (H)
    H = T + V
    delta_H = np.diff(H, axis=0, prepend=H[0:1])
    
    print("[*] Hunting the Coiled Traps (High V, Low T, \u0394H > 0)...")
    
    found_rupture = False
    
    # Start looking after the topology stabilizes
    for i in range(window + 10, len(sensorium)):
        
        curr_V = V[i]
        curr_T = T[i]
        curr_dH = delta_H[i]
        
        # Dynamic thresholds based on the current state of the entire manifold
        v_thresh = np.percentile(curr_V, 95)
        t_thresh = np.percentile(curr_T, 10)
        
        # The Physics Trap: Highly stretched from its peers, frozen, absorbing energy
        is_coiled = (curr_V > v_thresh) & (curr_T < t_thresh) & (curr_dH > 0)
        
        if is_coiled.any():
            trap_idx = np.where(is_coiled)[0][0]
            trap_sym = symbols[trap_idx]
            
            # Find the strongest peer pulling on it
            v_window = velocity[i-window:i]
            corr_matrix = np.corrcoef(v_window.T)
            np.fill_diagonal(corr_matrix, 0)
            peer_idx = np.argmax(corr_matrix[trap_idx])
            peer_sym = symbols[peer_idx]
            
            # Recalculate displacement direction
            adj_matrix = np.clip(corr_matrix, 0, None)
            weights_sum = np.sum(adj_matrix, axis=1) + 1e-9
            network_equilibrium = np.dot(adj_matrix, prices[i]) / weights_sum
            
            disp_val = prices[i, trap_idx] - network_equilibrium[trap_idx]
            
            ts = sensorium['chronological_time'].iloc[i].strftime('%Y-%m-%d %H:%M:%S')
            
            print("\n" + "="*85)
            print(f"AION CORE SOLVER  |  T=0: {ts}")
            print("="*85)
            
            print("[ WHAT IS THE MARKET DOING? ]")
            print(f"Phase State: TOPOLOGICAL DEFORMATION.")
            print(f"Energy is flowing through the manifold. Kinetic energy is expanding in local clusters.")
            
            print("\n[ WHY IS IT DOING IT? ]")
            print(f"Hidden Cause: [{peer_sym}] and its correlated cluster are moving violently.")
            print(f"The energy transfer is stretching the structural tethers of the network.")
            
            print("\n[ HOW DO WE GAIN FROM IT? ]")
            print(f"Trap Detected: [{trap_sym}] is mathematically tethered to the moving cluster.")
            print(f"Physics: [{trap_sym}] has Extreme Potential Energy (V: {curr_V[trap_idx]:.4f}).")
            print(f"         It is absorbing system energy (\u0394H > 0) but its Kinetic Energy is near-zero (T: {curr_T[trap_idx]:.6f}).")
            print(f"         It is acting as a capacitor. The structural tether is stretched to the breaking point.")
            
            print("\n[ ACTION PHYSICS ARBITER ]")
            direction = "SHORT (SELL)" if disp_val > 0 else "LONG (BUY)"
            print(f"Weakest Outlet: The forced geometric snap of [{trap_sym}] toward its Network Equilibrium.")
            print(f"Verdict: EXECUTE ASYMMETRIC ROUTE -> {direction} [{trap_sym}].")
            print("="*85)
            
            found_rupture = True
            break
            
    if not found_rupture:
        print("[-] Manifold is currently in total equilibrium. No coiled traps detected.")

if __name__ == "__main__":
    solve_the_problem()
