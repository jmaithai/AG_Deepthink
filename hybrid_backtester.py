import pandas as pd
import numpy as np
import os
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import physics_engine as pe
import kinematics_engine as ke
import policy_engine as po

def run_empirical_backtest():
    file_path = "archive/manifold_6M.parquet"
    print(f"[*] Loading Empirical History: {file_path}")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Cannot run backtest. Data not found.")
        return
        
    print("[*] Applying Hybrid Filter (Forex + Metals + Energies)...")
    valid_symbols = {
        'EURUSD', 'USDCHF', 'USDCAD', 'AUDUSD', 'GBPUSD', 'NZDUSD', 'USDJPY',
        'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'CADCHF', 'CADJPY', 'CHFJPY',
        'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD', 'GBPAUD',
        'GBPCAD', 'GBPJPY', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'NZDJPY',
        'XAUUSD', 'XAGUSD', 'WTI', 'BRENT', 'NATGAS'
    }
    
    # Filter columns to only include the 32 valid hybrid pairs
    vol_cols = [col for col in df.columns if col.endswith('_vol') and col.replace('_vol', '') in valid_symbols]
    price_cols = [col for col in df.columns if col.endswith('_price') and col.replace('_price', '') in valid_symbols]
    close_cols = [col for col in df.columns if col.endswith('_close') and col.replace('_close', '') in valid_symbols]
    
    print("[*] Warping Chronological Time into Thermodynamic States (Tau) using Hybrid Volume...")
    df['network_kinetic_energy'] = df[vol_cols].sum(axis=1)
    df['cumulative_energy'] = df['network_kinetic_energy'].cumsum()
    
    # Use standard energy threshold (25000)
    df['Energy_State'] = (df['cumulative_energy'] // pe.ENERGY_THRESHOLD).astype(int)
    
    thermo_df = df.groupby('Energy_State')[price_cols + close_cols].last()
    print(f"[+] Extracted {len(thermo_df)} Proper Time states (Hybrid Manifold).")
    
    active_edges = []
    active_nodes = set()
    for col in price_cols:
        sym = col.replace('_price', '')
        base, quote = sym[:3], sym[3:6]
        active_edges.append({'symbol': sym, 'base': base, 'quote': quote})
        active_nodes.update([base, quote])
        
    active_nodes = sorted(list(active_nodes))
    
    print("[*] Inverting Gauge (Extracting Hoberman Pressures)...")
    pressure_df = pe.extract_hoberman_space(thermo_df[price_cols], active_nodes, active_edges)
    
    print("[*] Calculating Kinematics & Volumetric Density (ATR Proxy)...")
    v, m, q, J, dJ, S_norm, phi = ke.compute_kinematics(pressure_df)
    
    atr_dict = {}
    for edge in active_edges:
        sym = edge['symbol']
        rets = thermo_df[f"{sym}_price"].diff().fillna(0)
        atr_dict[sym] = rets.rolling(21).std().fillna(0.001).values
    
    print("[*] Igniting PyTorch Simulation (HYBRID FX/COMMODITY THERMODYNAMICS)...")
    virtual_equity = 1000.0
    peak_equity = 1000.0
    max_drawdown = 0.0
    wins = 0
    losses = 0
    
    AI_STEP = 10 
    portfolio = {} 
    
    # In the real world, the daemon NEVER wipes its memory. 
    # It continuously trains on its persistent world_model.pt weights.
    for i in range(50, len(pressure_df)):
        step_pnl = 0.0
        ejected_positions = []
        current_phi = phi[i].item()
        
        for sym, pos in portfolio.items():
            current_price = thermo_df.iloc[i][f"{sym}_close"]
            pct_return = (current_price - pos['entry_price']) / pos['entry_price']
            if pos['direction'] == 'SHORT':
                pct_return = -pct_return
            
            current_atr = atr_dict[sym][i] + 1e-5
            
            # PnL based purely on exact notional sizing (like MT5 lots), no artificial multipliers
            pnl_contribution = pos['size'] * pct_return
            step_pnl += pnl_contribution
            
            pos['accumulated_pnl'] = pos.get('accumulated_pnl', 0.0) + pnl_contribution
            
            if current_phi < 0.5:
                ejected_positions.append((sym, pos['accumulated_pnl']))
            
        for sym, pnl in ejected_positions:
            if pnl > 0: wins += 1
            else: losses += 1
            del portfolio[sym]
            
        virtual_equity += step_pnl
        
        if virtual_equity > peak_equity:
            peak_equity = virtual_equity
        dd = (peak_equity - virtual_equity) / peak_equity
        if dd > max_drawdown:
            max_drawdown = dd
            
        for sym in portfolio:
            portfolio[sym]['entry_price'] = thermo_df.iloc[i][f"{sym}_close"]
            
        if i % AI_STEP == 0:
            for sym, pos in portfolio.items():
                if pos.get('accumulated_pnl', 0.0) > 0: wins += 1
                else: losses += 1
            portfolio.clear() 
            
            start_idx = max(0, i - 120)
            t_v, t_m, t_q = v[start_idx:i+1], m[start_idx:i+1], q[start_idx:i+1]
            t_J, t_dJ, t_phi = J[start_idx:i+1], dJ[start_idx:i+1], phi[start_idx:i+1]
            
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                final_actions, _ = po.decide_active_inference(active_nodes, active_edges, t_v, t_m, t_q, t_J, t_dJ, t_phi)
            except Exception:
                final_actions = None
            finally:
                sys.stdout.close()
                sys.stdout = old_stdout
                
            if current_phi >= 0.5 and final_actions is not None:
                total_conviction = np.sum(np.abs(final_actions)) + 1e-6
                for idx, edge in enumerate(active_edges):
                    sym = edge['symbol']
                    u_risk = final_actions[idx]
                    if abs(u_risk) > 0.05: 
                        weight = abs(u_risk) / total_conviction
                        # 10,000 notional units (exactly 0.10 standard MT5 lots) scaled by structural weight
                        allocation = 10000.0 * weight 
                        portfolio[sym] = {
                            'direction': 'LONG' if u_risk > 0 else 'SHORT',
                            'size': allocation,
                            'entry_price': thermo_df.iloc[i][f"{sym}_close"]
                        }
                        
            if i % (AI_STEP * 5) == 0:
                print(f"  -> [State {i}/{len(pressure_df)}] Equity: ${virtual_equity:.2f} | MDD: {max_drawdown*100:.2f}% | Active Pos: {len(portfolio)}")

    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    print("\n" + "="*50)
    print("EMPIRICAL 6-MONTH HYBRID MANIFOLD BACKTEST (OOS)")
    print("="*50)
    print(f"Final Equity:   ${virtual_equity:.2f} (Starting: $1000.00)")
    print(f"Total Return:   {((virtual_equity - 1000)/1000)*100:.2f}%")
    print(f"Max Drawdown:   {max_drawdown*100:.2f}%")
    print(f"Total Trades:   {total_trades}")
    print(f"Win Rate:       {win_rate:.2f}%")
    
    returns = ((virtual_equity - 1000)/1000)
    sharpe = (returns - 0.02) / (max_drawdown + 1e-9)
    print(f"Sharpe (Est):   {sharpe:.2f}")
    print("="*50)

if __name__ == "__main__":
    run_empirical_backtest()
