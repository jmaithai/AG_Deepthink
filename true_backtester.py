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
        print(f"[!] Cannot run backtest. Data not found. Run true_data_loader.py first.")
        return
        
    print("[*] Warping Chronological Time into Thermodynamic States (\u03C4)...")
    vol_cols = [col for col in df.columns if col.endswith('_vol')]
    df['network_kinetic_energy'] = df[vol_cols].sum(axis=1)
    df['cumulative_energy'] = df['network_kinetic_energy'].cumsum()
    df['Energy_State'] = (df['cumulative_energy'] // pe.ENERGY_THRESHOLD).astype(int)
    
    price_cols = [col for col in df.columns if col.endswith('_price')]
    close_cols = [col for col in df.columns if col.endswith('_close')]
    
    thermo_df = df.groupby('Energy_State')[price_cols + close_cols].last()
    print(f"[+] Extracted {len(thermo_df)} Proper Time states.")
    
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
        # Physical Volatility: Rolling standard deviation of log returns
        rets = thermo_df[f"{sym}_price"].diff().fillna(0)
        atr_dict[sym] = rets.rolling(21).std().fillna(0.001).values
    
    print("[*] Igniting PyTorch Simulation (V3: PURE THERMODYNAMICS)...")
    virtual_equity = 1000.0
    peak_equity = 1000.0
    max_drawdown = 0.0
    
    AI_STEP = 10 # Restored the high-resolution AI tracking
    portfolio = {} 
    
    # Pre-clear the world model so the backtest starts fresh
    if os.path.exists("world_model.pt"): os.remove("world_model.pt")
    if os.path.exists("last_action.pt"): os.remove("last_action.pt")
        
    for i in range(50, len(pressure_df)):
        
        step_pnl = 0.0
        ejected_positions = []
        current_phi = phi[i].item()
        
        # 1. Continuous PnL Update & KINEMATIC EJECTION
        for sym, pos in portfolio.items():
            current_price = thermo_df.iloc[i][f"{sym}_close"]
            pct_return = (current_price - pos['entry_price']) / pos['entry_price']
            if pos['direction'] == 'SHORT':
                pct_return = -pct_return
            
            # --- LAYER 4 EXECUTION MATH: VOLUMETRIC SCALING ---
            # Maps exactly to live execution logic (Risk / (2.5 * ATR))
            # Gold and Fiat are now scaled perfectly by their true kinetic density
            current_atr = atr_dict[sym][i] + 1e-5
            sim_return_multiplier = 1.0 / (2.5 * current_atr)
            
            step_pnl += pos['size'] * pct_return * sim_return_multiplier
            
            # --- THE VOLCANIC EJECTION (Pure Physics) ---
            # If the Volcano Gate closes, the system physically steps off the board immediately.
            if current_phi < 0.5:
                ejected_positions.append(sym)
            
        for sym in ejected_positions:
            del portfolio[sym]
            
        virtual_equity += step_pnl
        
        if virtual_equity > peak_equity:
            peak_equity = virtual_equity
        dd = (peak_equity - virtual_equity) / peak_equity
        if dd > max_drawdown:
            max_drawdown = dd
            
        # Update entry prices
        for sym in portfolio:
            portfolio[sym]['entry_price'] = thermo_df.iloc[i][f"{sym}_close"]
            
        # 2. AI Rebalancing Step 
        if i % AI_STEP == 0:
            portfolio.clear() 
            
            # --- THE LYAPUNOV HORIZON (Memory Decay) ---
            # Bounding the LSTM lookback window. Energy dissipates over time.
            # We feed the PyTorch brain only the coherent fluid cycle (last 120 states).
            # Eliminates the O(N^2) memory leak with physics, not clocks.
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
                        # Restoring the pure 15% energy budget target
                        allocation = virtual_equity * 0.15 * weight 
                        
                        portfolio[sym] = {
                            'direction': 'LONG' if u_risk > 0 else 'SHORT',
                            'size': allocation,
                            'entry_price': thermo_df.iloc[i][f"{sym}_close"]
                        }
                        
            # Print cleanly every 50 states
            if i % (AI_STEP * 5) == 0:
                print(f"  -> [State {i}/{len(pressure_df)}] Equity: ${virtual_equity:.2f} | MDD: {max_drawdown*100:.2f}% | Active Pos: {len(portfolio)}")

    print("\n" + "="*50)
    print("EMPIRICAL 6-MONTH TRUE BACKTEST (PURE PHYSICS)")
    print("="*50)
    print(f"Final Equity:   ${virtual_equity:.2f} (Starting: $1000.00)")
    print(f"Total Return:   {((virtual_equity - 1000)/1000)*100:.2f}%")
    print(f"Max Drawdown:   {max_drawdown*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_empirical_backtest()
