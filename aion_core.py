import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def self_assemble_manifold():
    """Dynamically builds the living manifold without human lists."""
    print("[*] AION SENSORIUM: Initiating Autopoietic Assembly...")
    if not mt5.initialize():
        print("[-] MT5 Init Failed.")
        return None, None
        
    all_symbols = mt5.symbols_get()
    if all_symbols is None:
        return None, None

    # 1. The Semantic Purge (Filter out structural noise)
    noise_strings = ['BTC', 'ETH', 'LTC', 'XRP', 'ADA', 'SOL', 'DOT', 'DOGE', 'UNI', 'SUI', 'SHIB']
    candidate_symbols = []
    
    for sym in all_symbols:
        name = sym.name.upper()
        # Drop crypto, drop hidden/disabled symbols
        if sym.visible and not any(noise in name for noise in noise_strings):
            candidate_symbols.append(sym.name)
            
    print(f"[*] Found {len(candidate_symbols)} candidate nodes. Testing for kinetic life...")
    
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24) # 24-hour lookback for life test
    
    raw_streams = []
    living_nodes = []
    
    # 2. The Kinetic Purge (Filter out dead/illiquid nodes)
    for sym in candidate_symbols:
        mt5.symbol_select(sym, True)
        # Pull M1 bars to quickly test liquidity presence
        rates = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M1, start_time, end_time)
        
        # A liquid fiat/index node should have roughly 1,440 M1 bars in 24 hours.
        # If it has less than 500, it is dead, illiquid, or a broken broker feed.
        if rates is not None and len(rates) > 500:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            temp_df = pd.DataFrame(index=df.index)
            temp_df[f"{sym}_mid"] = df['close'] # Using close as proxy for mid-price
            temp_df[f"{sym}_vol"] = df['tick_volume']
            
            raw_streams.append(temp_df)
            living_nodes.append(sym)
            
    mt5.shutdown()
    
    if not raw_streams:
        print("[-] Fatal: Manifold is entirely dead.")
        return None, None
        
    print(f"[+] Autopoiesis Complete. Manifold self-assembled at {len(living_nodes)} Dimensions.")
    
    print("[*] Fusing chronological tensor...")
    master_df = pd.concat(raw_streams, axis=1, join='outer')
    master_df.ffill(inplace=True)
    master_df.dropna(inplace=True)
    
    return master_df, living_nodes

def run_aion_core():
    df, symbols = self_assemble_manifold()
    if df is None: return
    
    print("\n[*] AION CORE SOLVER: Eradicating Chronological Time...")
    vol_cols = [c for c in df.columns if c.endswith('_vol')]
    df['network_ticks'] = df[vol_cols].sum(axis=1)
    df['cumulative_ticks'] = df['network_ticks'].cumsum()
    
    # Event Threshold scales dynamically to network density
    EVENT_THRESHOLD = int(df['network_ticks'].mean() * 100) 
    df['Event_Clock'] = (df['cumulative_ticks'] // EVENT_THRESHOLD).astype(int)
    
    close_cols = [f"{sym}_mid" for sym in symbols]
    
    df['chronological_time'] = df.index
    sensorium = df.groupby('Event_Clock').last()
    
    print(f"[*] Compressed into {len(sensorium):,} pure thermodynamic states.")

    # THE FIELD (Velocity & Covariance)
    prices = np.log(sensorium[close_cols].values + 1e-9)
    velocity = np.diff(prices, axis=0, prepend=prices[0:1])

    print("[*] Scanning Manifold for Reality Degeneracy and Tension Traps...")
    
    window = 21
    prev_entropy = 0
    found_rupture = False
    
    for i in range(window, len(sensorium)):
        v_window = velocity[i-window:i]
        
        # Normalize to find pure structural shape
        mean_v = np.mean(v_window, axis=0)
        std_v = np.std(v_window, axis=0) + 1e-12
        v_norm = (v_window[-1] - mean_v) / std_v # Current state normalized
        
        v_window_norm = (v_window - mean_v) / std_v
        cov_matrix = np.cov(v_window_norm.T)
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eig_vals)[::-1]
        
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]
        
        # Reality Degeneracy (Entropy)
        total_energy = np.sum(eig_vals) + 1e-12
        p_energy = np.clip(eig_vals / total_energy, 1e-12, 1.0)
        entropy = -np.sum(p_energy * np.log2(p_energy))
        
        # 3. PHASE TRANSITION TRIGGER
        if i > window:
            entropy_drop = entropy - prev_entropy
            
            # Trigger: Entropy collapses AND a single force dominates
            if entropy_drop < -0.20 and p_energy[0] > 0.40:
                
                dom_force = eig_vecs[:, 0]
                power = p_energy[0] * 100
                
                # How hard is the Manifold pushing right now?
                force_magnitude = np.dot(v_norm, dom_force)
                
                # What the Manifold DEMANDS each node to do
                expected_v = force_magnitude * dom_force
                
                # THE TRAP (Tension Discrepancy = Expected - Actual)
                tension_discrepancy = expected_v - v_norm
                
                trap_idx = np.argmax(np.abs(tension_discrepancy))
                prop_idx = np.argmax(np.abs(expected_v)) # Epicenter
                
                trap_sym = symbols[trap_idx]
                prop_sym = symbols[prop_idx]
                
                ts = sensorium['chronological_time'].iloc[i].strftime('%Y-%m-%d %H:%M:%S')
                
                print("\n" + "="*85)
                print(f"AION LIVING THESIS  |  T=0: {ts}")
                print("="*85)
                
                print("[ WHAT IS THE MARKET DOING? ]")
                print(f"Phase State: FRACTURE & COMPRESSION.")
                print(f"Reality Degeneracy collapsed by {abs(entropy_drop):.4f} bits. (Current Entropy: {entropy:.2f})")
                
                print("\n[ WHY IS IT DOING IT? ]")
                print(f"Hidden Cause: A singular structural force commands {power:.1f}% of the {len(symbols)}-D manifold.")
                print(f"Propagation: The pressure wave is visibly anchored by [{prop_sym}].")
                
                print("\n[ HOW DO WE GAIN FROM IT? ]")
                trap_exp = expected_v[trap_idx]
                trap_act = v_norm[trap_idx]
                
                print(f"Trap Detected: [{trap_sym}] has massive Tension Discrepancy.")
                print(f"Physics: The {len(symbols)}-D Manifold demands [{trap_sym}] move with velocity {trap_exp:+.2f}.")
                print(f"         But local participants are holding it at {trap_act:+.2f}.")
                print(f"         Retail is fighting the global Eigen-force. They are fatally trapped.")
                
                print("\n[ ACTION PHYSICS ARBITER ]")
                direction = "LONG (BUY)" if trap_exp > 0 else "SHORT (SELL)"
                print(f"Weakest Outlet: The forced liquidation of [{trap_sym}] participants.")
                print(f"Verdict: TRAP IDENTIFIED. EXECUTE FORCED RELEASE -> {direction} [{trap_sym}].")
                print("="*85)
                
                found_rupture = True
                break # Show first major rupture to prove the code
                
        prev_entropy = entropy
        
    if not found_rupture:
        print("\n[-] Manifold is currently stable. No structural ruptures detected in the last 24 hours.")

if __name__ == "__main__":
    run_aion_core()
