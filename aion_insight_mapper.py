import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_insight_mapper():
    file_path = "archive/fractal_manifold_6M.parquet"
    print("[*] AION INSIGHT MAPPER: Eradicating Martingale Logic. Seeking True Physics...")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Error: {e}")
        return

    # Event Clock based on M1 physical kinetic injection
    vol_cols = [c for c in df.columns if c.endswith('_M1_vol')]
    df['network_ticks'] = df[vol_cols].sum(axis=1)
    df['cumulative_ticks'] = df['network_ticks'].cumsum()
    EVENT_THRESHOLD = 25000 
    df['Event_Clock'] = (df['cumulative_ticks'] // EVENT_THRESHOLD).astype(int)
    
    # Gauge Inversion Fix: M1 suffix is _M1_price
    close_cols_m1 = [c for c in df.columns if c.endswith('_M1_price')]
    close_cols_m15 = [c for c in df.columns if c.endswith('_M15_price')]
    symbols = [c.replace('_M1_price', '') for c in close_cols_m1]
    
    df['chronological_time'] = df.index
    sensorium = df.groupby('Event_Clock').last()

    print(f"[*] Analyzing {len(sensorium):,} Event States for Fractal Discrepancies...")

    # Macro/Meso Tide (M15)
    # Gauge Inversion Fix: Values are already log-transformed
    prices_m15 = sensorium[close_cols_m15].values
    vel_m15 = np.diff(prices_m15, axis=0, prepend=prices_m15[0:1])

    # Micro Flow (M1)
    # Gauge Inversion Fix: Values are already log-transformed
    prices_m1 = sensorium[close_cols_m1].values
    vel_m1 = np.diff(prices_m1, axis=0, prepend=prices_m1[0:1])
    vol_m1 = sensorium[vol_cols].values

    window = 21
    
    for i in range(window, len(sensorium)):
        # 1. THE MACRO TIDE (What the manifold is doing)
        v_window_m15 = vel_m15[i-window:i]
        v_norm_m15 = (v_window_m15 - np.mean(v_window_m15, axis=0)) / (np.std(v_window_m15, axis=0) + 1e-12)
        
        cov_matrix = np.cov(v_norm_m15.T)
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eig_vals)[::-1]
        
        dom_force_m15 = eig_vecs[:, idx[0]]
        force_magnitude = np.dot(v_norm_m15[-1], dom_force_m15)
        
        # The true structural pull on each node
        macro_tide = force_magnitude * dom_force_m15
        
        # 2. THE MICRO RESISTANCE (What the trapped participants are doing)
        # Work = Force (Velocity) * Mass (Volume)
        micro_work = vel_m1[i] * vol_m1[i]
        
        # 3. TRAP DETECTION (The Discrepancy)
        # We multiply the Macro Tide by the Micro Work.
        # If Macro is UP (+) and Micro is heavily DOWN (-), the product is deeply NEGATIVE.
        # This means massive volume is burning capital fighting the global wave.
        
        trap_matrix = macro_tide * micro_work
        
        # We look for the most negative value (The deepest trap)
        deepest_trap_val = np.min(trap_matrix)
        
        if deepest_trap_val < -np.percentile(np.abs(trap_matrix), 95): # Extreme fractal divergence
            trap_idx = np.argmin(trap_matrix)
            trap_sym = symbols[trap_idx]
            
            macro_dir = "BULLISH (UP)" if macro_tide[trap_idx] > 0 else "BEARISH (DOWN)"
            micro_dir = "SELLING (DOWN)" if vel_m1[i][trap_idx] < 0 else "BUYING (UP)"
            
            ts = sensorium['chronological_time'].iloc[i].strftime('%Y-%m-%d %H:%M:%S')
            
            print("\n" + "="*85)
            print(f"AION INSIGHT THESIS  |  T=0: {ts}")
            print("="*85)
            
            print("[ WHAT IS THE MARKET DOING? ]")
            print(f"Fractal Divergence: M15 Macro Tide is violently colliding with M1 Micro Flow.")
            
            print("\n[ WHY IS IT DOING IT? ]")
            print(f"Hidden Cause: The global manifold is exerting a {macro_dir} structural pull on [{trap_sym}].")
            print(f"The Trap: Local participants are stubbornly {micro_dir} [{trap_sym}], injecting massive volume against the tide.")
            
            print("\n[ HOW DO WE GAIN FROM IT? ]")
            print(f"Physics: The micro-participants are burning capital to hold the dam.")
            print(f"         They cannot sustain the volume against the M15 tectonic plate.")
            print(f"         Their stop-losses and margin calls are located precisely where the macro wave wants to go.")
            
            print("\n[ ACTION PHYSICS ARBITER ]")
            trade_dir = "LONG (BUY)" if macro_tide[trap_idx] > 0 else "SHORT (SELL)"
            print(f"Weakest Outlet: The forced liquidation of the M1 participants on [{trap_sym}].")
            print(f"Verdict: WAIT for M1 velocity to stall, then execute {trade_dir} to harvest the snap.")
            print("="*85)
            
            break # Print the first clear insight to prove the logic

if __name__ == "__main__":
    run_insight_mapper()
