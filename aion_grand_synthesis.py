import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_grand_synthesis():
    file_path = "archive/fractal_manifold_6M.parquet"
    print(f"[*] AION GRAND SYNTHESIS: Entropy Phase Transition + Barycentric Trap...")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Error: {e}")
        return
        
    print("[*] Applying Doctrine Firewall: Filtering non-tradable nodes...")
    valid_symbols = {
        'EURUSD', 'USDCHF', 'USDCAD', 'AUDUSD', 'GBPUSD', 'NZDUSD', 'USDJPY',
        'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'CADCHF', 'CADJPY', 'CHFJPY',
        'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD', 'GBPAUD',
        'GBPCAD', 'GBPJPY', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'NZDJPY',
        'XAUUSD', 'XAGUSD', 'WTI', 'BRENT', 'NATGAS'
    }
    
    close_cols = [c for c in df.columns if c.endswith('_M1_close') and c.replace('_M1_close', '') in valid_symbols]
    vol_cols = [c.replace('_M1_close', '_M1_vol') for c in close_cols]
    
    df['network_ticks'] = df[vol_cols].sum(axis=1)
    if df['network_ticks'].sum() == 0:
        df['network_ticks'] = 1.0
        
    df['cumulative_ticks'] = df['network_ticks'].cumsum()
    EVENT_THRESHOLD = 25000 
    df['Event_Clock'] = (df['cumulative_ticks'] // EVENT_THRESHOLD).astype(int)
    symbols = [c.replace('_M1_close', '') for c in close_cols]
    
    df['chronological_time'] = df.index
    sensorium = df.groupby('Event_Clock').last()
    
    print(f"[+] Compressed into {len(sensorium):,} True Event States.")

    raw_prices = sensorium[close_cols].values
    prices = (raw_prices - np.mean(raw_prices, axis=0)) / (np.std(raw_prices, axis=0) + 1e-9)
    volumes = sensorium[vol_cols].values
    mass = (volumes - np.mean(volumes, axis=0)) / (np.std(volumes, axis=0) + 1e-9)
    mass = np.clip(mass + 1.0, 0.1, None) 
    
    velocity = np.diff(prices, axis=0, prepend=prices[0:1])
    barycenter = np.sum(mass * prices, axis=1) / (np.sum(mass, axis=1) + 1e-9)
    
    T = 0.5 * mass * (velocity**2) 
    displacement = prices - barycenter[:, None]
    V = 0.5 * mass * (displacement**2) 
    H = T + V 
    delta_H = np.diff(H, axis=0, prepend=H[0:1])

    passes = 0
    failures = 0
    look_forward = 20 
    window = 21
    prev_entropy = 0
    total_phase_transitions = 0
    
    print("[*] Executing Deep Sweep: Dual-Confirmation Physics...")
    
    for i in range(window, len(sensorium) - look_forward):
        # 1. ENTROPY CALCULATION (Phase Transition)
        v_window = velocity[i-window:i]
        mean_v = np.mean(v_window, axis=0)
        std_v = np.std(v_window, axis=0) + 1e-12
        v_window_norm = (v_window - mean_v) / std_v
        
        cov_matrix = np.cov(v_window_norm.T)
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[idx]
        
        total_energy = np.sum(eig_vals) + 1e-12
        p_energy = np.clip(eig_vals / total_energy, 1e-12, 1.0)
        entropy = -np.sum(p_energy * np.log2(p_energy))
        
        if i > window:
            entropy_drop = entropy - prev_entropy
            
            # TRIGGER 1: Systemic Phase Transition (The Wave)
            if entropy_drop < -0.15 and p_energy[0] > 0.40:
                total_phase_transitions += 1
                
                # TRIGGER 2: The Coiled Trap (The Anchor)
                current_V = V[i]
                current_T = T[i]
                current_dH = delta_H[i]
                
                v_thresh = np.percentile(current_V, 90)
                t_thresh = np.percentile(current_T, 10)
                
                is_coiled = (current_V > v_thresh) & (current_T < t_thresh) & (current_dH > 0)
                
                if is_coiled.any():
                    for idx_node in np.where(is_coiled)[0]:
                        disp_t0 = abs(displacement[i, idx_node])
                        
                        future_disp = np.abs(displacement[i+1 : i+1+look_forward, idx_node])
                        future_T = T[i+1 : i+1+look_forward, idx_node]
                        
                        min_future_disp = np.min(future_disp)
                        max_future_T = np.max(future_T)
                        
                        if min_future_disp < disp_t0 and max_future_T > current_T[idx_node]:
                            passes += 1
                        else:
                            failures += 1
                            
        prev_entropy = entropy

    win_rate = (passes / (passes + failures)) * 100 if (passes + failures) > 0 else 0
    
    print("\n" + "="*80)
    print("AION GRAND SYNTHESIS (ENTROPY + BARYCENTER) OOS AUDIT")
    print("="*80)
    print(f"Total Global Phase Transitions Detected: {total_phase_transitions}")
    print(f"Total Grand Synthesis Traps Executed:    {passes + failures}")
    print(f"Mathematical Win Rate (Forced Release):  {win_rate:.2f}%")
    print(f"Successful Snaps:                        {passes}")
    print(f"Physics Failures:                        {failures}")
    print("="*80)

if __name__ == "__main__":
    run_grand_synthesis()
