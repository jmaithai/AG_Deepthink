import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test_kinetic_propagation():
    file_path = "archive/fractal_manifold_6M.parquet"
    print("[*] KINETIC PROPAGATION AUDIT: Pure Energy Surfing...")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Error: {e}")
        return
        
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
    df['cumulative_ticks'] = df['network_ticks'].cumsum()
    EVENT_THRESHOLD = 25000 
    df['Event_Clock'] = (df['cumulative_ticks'] // EVENT_THRESHOLD).astype(int)
    
    sensorium = df.groupby('Event_Clock').last()

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
    delta_T = np.diff(T, axis=0, prepend=T[0:1]) # Acceleration of Kinetic Energy

    # ---------------------------------------------------------
    # KINETIC PROPAGATION SIMULATION
    # ---------------------------------------------------------
    active_surfers = {} # {idx: {'direction': 1 or -1, 'entry_price_norm': float}}
    
    total_liquidations = 0
    total_raw_return = 0.0
    wins = 0
    losses = 0
    equity_curve = []

    for i in range(10, len(sensorium)):
        curr_T = T[i]
        curr_dT = delta_H[i] # Use Enthalpy for global energy injection check
        curr_accel_T = delta_T[i]
        curr_price = prices[i]
        curr_vel = velocity[i]
        
        t_thresh = np.percentile(curr_T, 90)
        
        # 1. Evaluate Existing Surfers (Exhaustion Check)
        ejected_surfers = []
        for idx, pos_data in active_surfers.items():
            # If the Kinetic Acceleration drops to 0 or goes negative, the fuel is gone. Exit instantly.
            if curr_accel_T[idx] <= 0:
                ejected_surfers.append(idx)
                
        for idx in ejected_surfers:
            exit_price_norm = curr_price[idx]
            trade_pnl = (exit_price_norm - active_surfers[idx]['entry_price_norm']) * active_surfers[idx]['direction']
            total_raw_return += trade_pnl
            total_liquidations += 1
            
            if trade_pnl > 0: wins += 1
            else: losses += 1
            del active_surfers[idx]

        # 2. Evaluate Manifold for Epicenters (Kinetic Injection)
        # We want nodes where T is massive (top 10%), accelerating (dT > 0), and energy is expanding (dH > 0)
        is_epicenter = (curr_T > t_thresh) & (curr_accel_T > 0) & (curr_dT > 0)
        
        if is_epicenter.any():
            for idx in np.where(is_epicenter)[0]:
                if idx not in active_surfers:
                    # Ride the wave in the direction of the velocity
                    direction = 1 if curr_vel[idx] > 0 else -1
                    active_surfers[idx] = {'direction': direction, 'entry_price_norm': curr_price[idx]}

        # Track floating equity
        floating_pnl = 0.0
        for idx, pos_data in active_surfers.items():
            trade_pnl = (curr_price[idx] - pos_data['entry_price_norm']) * pos_data['direction']
            floating_pnl += trade_pnl
            
        current_equity = total_raw_return + floating_pnl
        equity_curve.append(current_equity)

    # Close any remaining at the end
    for idx, pos_data in active_surfers.items():
        exit_price_norm = prices[-1, idx]
        trade_pnl = (exit_price_norm - pos_data['entry_price_norm']) * pos_data['direction']
        total_raw_return += trade_pnl
        total_liquidations += 1
        if trade_pnl > 0: wins += 1
        else: losses += 1

    win_rate = (wins / total_liquidations) * 100 if total_liquidations > 0 else 0
    
    equity_curve = np.array(equity_curve)
    high_water_marks = np.maximum.accumulate(equity_curve)
    drawdowns = high_water_marks - equity_curve
    max_drawdown = np.max(drawdowns)
    
    print("\n" + "="*80)
    print("AION PURE KINETIC PROPAGATION AUDIT (NO DRAWDOWN SCALING)")
    print("="*80)
    print(f"Total Liquidations Executed:     {total_liquidations:,}")
    print(f"Mathematical Win Rate:           {win_rate:.2f}%")
    print(f"Kinetic Exhaustion Snaps (Wins): {wins:,}")
    print(f"Whipsaw Failures (Losses):       {losses:,}")
    print(f"Max Tension Scale (Mass Added):  1x (Pure single bullet, no scaling)")
    print(f"Max Floating Drawdown:           -{max_drawdown:.2f} Z-Score units")
    print(f"Net System Normalized Return:    +{total_raw_return:.2f} Z-Score units")
    print("="*80)

if __name__ == "__main__":
    test_kinetic_propagation()
