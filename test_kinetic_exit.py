import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test_kinetic_exit():
    file_path = "archive/fractal_manifold_6M.parquet"
    print(f"[*] AION KINETIC EXIT SIMULATION: Testing Dynamic Reversion Harvest...")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Error: {e}")
        return
        
    print("[*] Applying Doctrine Firewall...")
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

    print("[*] Executing Kinetic Trailing Exit Simulation...")
    
    wins = 0
    losses = 0
    total_raw_return = 0.0
    
    # We must allow the physics maximum room to resolve
    look_forward = 20 
    
    for i in range(10, len(sensorium) - look_forward):
        current_V = V[i]
        current_T = T[i]
        current_dH = delta_H[i]
        
        v_thresh = np.percentile(current_V, 90)
        t_thresh = np.percentile(current_T, 10)
        
        is_coiled = (current_V > v_thresh) & (current_T < t_thresh) & (current_dH > 0)
        
        if is_coiled.any():
            for idx in np.where(is_coiled)[0]:
                entry_price_norm = prices[i, idx]
                entry_disp = displacement[i, idx]
                
                direction = -1 if entry_disp > 0 else 1 
                
                # True Forward-Walk Kinetic Trailing Stop
                # We simulate live execution. We step forward one Event State at a time.
                peak_disp = abs(entry_disp)
                peak_idx = i
                trade_closed = False
                
                for t_step in range(1, look_forward + 1):
                    future_idx = i + t_step
                    current_disp = abs(displacement[future_idx, idx])
                    current_v = abs(velocity[future_idx, idx])
                    
                    # Update peak if we move closer to the Barycenter
                    if current_disp < peak_disp:
                        peak_disp = current_disp
                        peak_idx = future_idx
                        
                    # Calculate how far we have drawn down from the peak profit
                    drawdown = current_disp - peak_disp
                    
                    # Exit Trigger: If we draw down more than 1.0 local velocity units from our peak profit
                    # AND we have actually made some profit (peak_disp < entry_disp)
                    if drawdown > (current_v * 1.0) and peak_disp < abs(entry_disp):
                        exit_price_norm = prices[future_idx, idx]
                        trade_pnl = (exit_price_norm - entry_price_norm) * direction
                        total_raw_return += trade_pnl
                        
                        if trade_pnl > 0:
                            wins += 1
                        else:
                            losses += 1
                            
                        trade_closed = True
                        break
                        
                # Time out (Tension ruptured)
                if not trade_closed:
                    exit_price_norm = prices[i + look_forward, idx]
                    trade_pnl = (exit_price_norm - entry_price_norm) * direction
                    total_raw_return += trade_pnl
                    if trade_pnl > 0:
                        wins += 1
                    else:
                        losses += 1

    total_trades = wins + losses
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    
    print("\n" + "="*80)
    print("AION TRADE MANAGEMENT (KINETIC EXIT) AUDIT")
    print("="*80)
    print(f"Total Live Executions simulated: {total_trades:,}")
    print(f"Mathematical Win Rate (Actual Exit): {win_rate:.2f}%")
    print(f"Barycenter Snaps Harvested:          {wins:,}")
    print(f"Physical Failures (Losses):          {losses:,}")
    print(f"Net System Normalized Return:        +{total_raw_return:.2f} Z-Score units")
    print("="*80)

if __name__ == "__main__":
    test_kinetic_exit()
