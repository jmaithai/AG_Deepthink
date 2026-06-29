import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test_thermodynamic_farm():
    file_path = "archive/fractal_manifold_6M.parquet"
    print(f"[*] AION FARM SIMULATION: Continuous State Execution (Scaling & Equilibrium)...")
    
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
    if df['network_ticks'].sum() == 0:
        df['network_ticks'] = 1.0
        
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

    # ---------------------------------------------------------
    # FARM SIMULATION LOGIC
    # ---------------------------------------------------------
    active_farms = {} # {idx: {'direction': 1 or -1, 'mass': 1, 'entry_price_norm': float}}
    
    total_liquidations = 0
    total_raw_return = 0.0
    wins = 0
    losses = 0
    max_mass = 0
    equity_curve = []

    for i in range(10, len(sensorium)):
        curr_V = V[i]
        curr_T = T[i]
        curr_dH = delta_H[i]
        curr_disp = displacement[i]
        curr_price = prices[i]
        
        v_thresh = np.percentile(curr_V, 90)
        t_thresh = np.percentile(curr_T, 10)
        
        # 1. Evaluate Existing Farms (Liquidation Check)
        ejected_farms = []
        for idx, pos_data in active_farms.items():
            node_disp = curr_disp[idx]
            
            # LONG = direction 1 (trapped below, so disp < 0). Liquidate if disp >= 0
            if pos_data['direction'] == 1 and node_disp >= 0:
                ejected_farms.append(idx)
            # SHORT = direction -1 (trapped above, so disp > 0). Liquidate if disp <= 0
            elif pos_data['direction'] == -1 and node_disp <= 0:
                ejected_farms.append(idx)
                
        for idx in ejected_farms:
            # Liquidate!
            exit_price_norm = curr_price[idx]
            entry_price_norm = active_farms[idx]['entry_price_norm']
            direction = active_farms[idx]['direction']
            mass_multiplier = active_farms[idx]['mass']
            
            # The PnL is the average distance covered * the accumulated mass
            trade_pnl = (exit_price_norm - entry_price_norm) * direction * mass_multiplier
            total_raw_return += trade_pnl
            total_liquidations += 1
            
            if trade_pnl > 0:
                wins += 1
            else:
                losses += 1
                
            del active_farms[idx]

        # 2. Evaluate Manifold for New/Scaling Tension
        is_coiled = (curr_V > v_thresh) & (curr_T < t_thresh) & (curr_dH > 0)
        
        if is_coiled.any():
            for idx in np.where(is_coiled)[0]:
                direction = -1 if curr_disp[idx] > 0 else 1
                
                if idx not in active_farms:
                    active_farms[idx] = {'direction': direction, 'mass': 1, 'entry_price_norm': curr_price[idx]}
                else:
                    # Prevent direction swap doubling
                    if active_farms[idx]['direction'] != direction:
                        # Force Liquidation due to physics swap
                        exit_price_norm = curr_price[idx]
                        trade_pnl = (exit_price_norm - active_farms[idx]['entry_price_norm']) * active_farms[idx]['direction'] * active_farms[idx]['mass']
                        total_raw_return += trade_pnl
                        total_liquidations += 1
                        if trade_pnl > 0: wins += 1
                        else: losses += 1
                        del active_farms[idx]
                        continue
                        
                    # Scale into the tension!
                    # We recalculate the average entry price based on the new mass
                    old_mass = active_farms[idx]['mass']
                    new_mass = old_mass + 1
                    avg_entry = ((active_farms[idx]['entry_price_norm'] * old_mass) + curr_price[idx]) / new_mass
                    
                    active_farms[idx]['mass'] = new_mass
                    active_farms[idx]['entry_price_norm'] = avg_entry
                    
                    if new_mass > max_mass:
                        max_mass = new_mass

        # Track floating equity
        floating_pnl = 0.0
        for idx, pos_data in active_farms.items():
            trade_pnl = (curr_price[idx] - pos_data['entry_price_norm']) * pos_data['direction'] * pos_data['mass']
            floating_pnl += trade_pnl
            
        current_equity = total_raw_return + floating_pnl
        equity_curve.append(current_equity)

    # Close any remaining farms at the end of the simulation
    for idx, pos_data in active_farms.items():
        exit_price_norm = prices[-1, idx]
        trade_pnl = (exit_price_norm - pos_data['entry_price_norm']) * pos_data['direction'] * pos_data['mass']
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
    print("AION TRADE MANAGEMENT (THERMODYNAMIC FARM) AUDIT")
    print("="*80)
    print(f"Total Liquidations Executed:     {total_liquidations:,}")
    print(f"Mathematical Win Rate:           {win_rate:.2f}%")
    print(f"Absolute Equilibrium Snaps:      {wins:,}")
    print(f"Physical Failures (Losses):      {losses:,}")
    print(f"Max Tension Scale (Mass Added):  {max_mass}x")
    print(f"Max Floating Drawdown:           -{max_drawdown:.2f} Z-Score units")
    print(f"Net System Normalized Return:    +{total_raw_return:.2f} Z-Score units")
    print("="*80)

if __name__ == "__main__":
    test_thermodynamic_farm()
