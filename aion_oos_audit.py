import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_oos_audit():
    # We must use the 6-month data archive, not the 48-hour one
    file_path = "archive/fractal_manifold_6M.parquet"
    print(f"[*] AION OOS AUDIT: Loading 6-Month Macro Manifold from {file_path}...")
    
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
        
    print("[*] Eradicating Chronological Time (Event Clock compression)...")
    df['network_ticks'] = df[vol_cols].sum(axis=1)
    
    if df['network_ticks'].sum() == 0:
        df['network_ticks'] = 1.0
        
    df['cumulative_ticks'] = df['network_ticks'].cumsum()
    
    # Use 50,000 ticks for consistency across M1 6M datasets
    EVENT_THRESHOLD = 50000 
    df['Event_Clock'] = (df['cumulative_ticks'] // EVENT_THRESHOLD).astype(int)
    
    symbols = [c.replace('_M1_close', '') for c in close_cols]
    
    df['chronological_time'] = df.index
    sensorium = df.groupby('Event_Clock').last()
    
    print(f"[+] Compressed into {len(sensorium):,} True Event States.")

    print("[*] Calculating Manifold Barycenter and Energy States...")
    raw_prices = sensorium[close_cols].values
    
    prices = (raw_prices - np.mean(raw_prices, axis=0)) / (np.std(raw_prices, axis=0) + 1e-9)
    volumes = sensorium[vol_cols].values
    mass = (volumes - np.mean(volumes, axis=0)) / (np.std(volumes, axis=0) + 1e-9)
    mass = np.clip(mass, 0.1, None) 
    
    velocity = np.diff(prices, axis=0, prepend=prices[0:1])
    
    barycenter = np.sum(mass * prices, axis=1) / (np.sum(mass, axis=1) + 1e-9)
    
    T = 0.5 * mass * (velocity**2) 
    displacement = prices - barycenter[:, None]
    V = 0.5 * mass * (displacement**2) 
    
    H = T + V 
    delta_H = np.diff(H, axis=0, prepend=H[0:1])

    print("[*] Executing Deep OOS Sweep & Failure Autopsy...")
    
    passes = 0
    failures = []
    
    # We look forward 20 Event States (Physical time) to verify the snap
    look_forward = 20 
    
    for i in range(10, len(sensorium) - look_forward):
        current_V = V[i]
        current_T = T[i]
        current_dH = delta_H[i]
        
        # Local threshold bounds
        v_thresh = np.percentile(current_V, 90)
        t_thresh = np.percentile(current_T, 10)
        
        # The Trap: Highly stretched, frozen, and actively absorbing energy
        is_coiled = (current_V > v_thresh) & (current_T < t_thresh) & (current_dH > 0)
        
        if is_coiled.any():
            for idx in np.where(is_coiled)[0]:
                sym = symbols[idx]
                disp_t0 = abs(displacement[i, idx])
                
                future_disp = np.abs(displacement[i+1 : i+1+look_forward, idx])
                future_T = T[i+1 : i+1+look_forward, idx]
                
                min_future_disp = np.min(future_disp)
                max_future_T = np.max(future_T)
                
                # FALSIFICATION: Did it snap back toward the Barycenter AND release kinetic energy?
                if min_future_disp < disp_t0 and max_future_T > current_T[idx]:
                    passes += 1
                else:
                    # --- FORENSIC AUTOPSY ---
                    barycenter_shift = abs(barycenter[i+look_forward] - barycenter[i])
                    avg_manifold_movement = np.mean(np.abs(np.diff(barycenter[i:i+look_forward])))
                    
                    future_dH = delta_H[i+1 : i+1+look_forward, idx]
                    
                    if barycenter_shift > avg_manifold_movement * 2:
                        reason = "BARYCENTER DRIFT (The market moved to the node)"
                    elif np.all(future_dH < 0):
                        reason = "ENERGY LEAK (Potential bled out without kinetic release)"
                    else:
                        reason = "STRUCTURAL RUPTURE (Node broke away from manifold)"
                        
                    failures.append({'State': i, 'Symbol': sym, 'Reason': reason})

    total_events = passes + len(failures)
    win_rate = (passes / total_events) * 100 if total_events > 0 else 0
    
    print("\n" + "="*80)
    print("AION 6-MONTH OOS MACRO AUDIT: RESULTS")
    print("="*80)
    print(f"Total Coiled Traps Analyzed: {total_events:,}")
    print(f"Mathematical Win Rate:       {win_rate:.2f}%")
    print(f"Successful Snaps:            {passes:,}")
    print(f"Physics Failures:            {len(failures):,}")
    print("-" * 80)
    
    if len(failures) > 0:
        fail_df = pd.DataFrame(failures)
        reasons = fail_df['Reason'].value_counts()
        print("[ FAILURE AUTOPSY ]")
        for reason, count in reasons.items():
            print(f"- {reason}: {count} occurrences ({(count/len(failures))*100:.1f}%)")
            
        print("\n[ TOP OFFENDING NODES (Most Failures) ]")
        print(fail_df['Symbol'].value_counts().head(5).to_string())
    print("="*80)

if __name__ == "__main__":
    run_oos_audit()
