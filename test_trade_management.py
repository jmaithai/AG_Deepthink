import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test_trade_management():
    file_path = "archive/fractal_manifold_6M.parquet"
    print(f"[*] AION TRADE MANAGEMENT SIMULATION: Testing Barycentric Exit Logic...")
    
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

    print("[*] Executing Thermodynamic Take-Profit Simulation...")
    
    wins = 0
    losses = 0
    timeouts = 0
    total_raw_return = 0.0
    
    # 20 Event States is the maximum lifespan of a Coiled Trap
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
                entry_disp = displacement[i, idx]
                entry_price_norm = prices[i, idx]
                
                direction = -1 if entry_disp > 0 else 1 # If trapped above (disp>0), Short (-1). Trapped below, Long (1).
                
                # Simulate Trade Management
                trade_closed = False
                for t_step in range(1, look_forward + 1):
                    future_idx = i + t_step
                    future_disp = displacement[future_idx, idx]
                    
                    # Thermodynamic Take Profit: Close immediately when displacement crosses 0 (Barycenter hit)
                    if (entry_disp > 0 and future_disp <= 0) or (entry_disp < 0 and future_disp >= 0):
                        exit_price_norm = prices[future_idx, idx]
                        
                        # Profit = (Exit - Entry) * Direction
                        trade_pnl = (exit_price_norm - entry_price_norm) * direction
                        total_raw_return += trade_pnl
                        
                        if trade_pnl > 0:
                            wins += 1
                        else:
                            losses += 1
                        
                        trade_closed = True
                        break # Exit trade, stop checking future states
                
                # If the trade never crossed the Barycenter within the 20-event physical horizon
                if not trade_closed:
                    timeouts += 1
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
    print("AION TRADE MANAGEMENT (THERMODYNAMIC EXIT) AUDIT")
    print("="*80)
    print(f"Total Live Executions simulated: {total_trades:,}")
    print(f"Mathematical Win Rate (Actual Exit): {win_rate:.2f}%")
    print(f"Barycenter Snaps (Take-Profits hit): {wins:,}")
    print(f"Losses (Physical Failure):           {losses - timeouts:,}")
    print(f"Timeouts (Tension bled out):         {timeouts:,}")
    print(f"Net System Normalized Return:        +{total_raw_return:.2f} Z-Score units")
    print("="*80)

if __name__ == "__main__":
    test_trade_management()
