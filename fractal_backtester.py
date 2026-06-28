import pandas as pd
import numpy as np
import os
import sys
import warnings
import torch

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import physics_engine as pe
import kinematics_engine as ke
import policy_engine as po

def extract_frequency_physics(df, active_nodes, active_edges, tf_label):
    """Isolates a specific timeframe and inverts its Hoberman Space."""
    cols = [f"{e['symbol']}_{tf_label}_price" for e in active_edges]
    sub_df = df[cols].copy()
    # Temporarily rename to standard for pe compatibility
    sub_df.columns = [c.replace(f"_{tf_label}_price", "_price") for c in sub_df.columns]
    
    P = pe.extract_hoberman_space(sub_df, active_nodes, active_edges)
    return ke.compute_kinematics(P)

def run_fractal_backtest():
    file_path = "archive/fractal_manifold_6M.parquet"
    print(f"[*] Loading Fractal Tensor: {file_path}")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Critical Error: Data missing. {e}")
        return
        
    print("[*] Warping Chronological Time into Thermodynamic States (\u03C4)...")
    # Base Proper Time off the M1 (Micro) kinetic energy injection
    vol_cols = [col for col in df.columns if col.endswith('_M1_vol')]
    df['network_kinetic_energy'] = df[vol_cols].sum(axis=1)
    df['cumulative_energy'] = df['network_kinetic_energy'].cumsum()
    df['Energy_State'] = (df['cumulative_energy'] // pe.ENERGY_THRESHOLD).astype(int)
    
    # We must keep the raw closes for true PnL tracking
    close_cols = [col for col in df.columns if col.endswith('_M1_close')]
    
    # Collapse into thermodynamic states
    thermo_df = df.groupby('Energy_State').last()
    print(f"[+] Extracted {len(thermo_df)} Fractal Proper Time states.")
    
    # Topology map
    active_edges = []
    active_nodes = set()
    for col in close_cols:
        sym = col.replace('_M1_close', '')
        base, quote = sym[:3], sym[3:6]
        active_edges.append({'symbol': sym, 'base': base, 'quote': quote})
        active_nodes.update([base, quote])
        
    active_nodes = sorted(list(active_nodes))
    
    print("[*] Computing Multi-Dimensional Physics (H1 and M1)...")
    v_H1, m_H1, q_H1, J_H1, dJ_H1, S_H1, phi_H1 = extract_frequency_physics(thermo_df, active_nodes, active_edges, "H1")
    v_M1, m_M1, q_M1, J_M1, dJ_M1, S_M1, phi_M1 = extract_frequency_physics(thermo_df, active_nodes, active_edges, "M1")
    
    print("[*] Igniting PyTorch Fractal Resonance Simulation (Pure 1:1 Yield)...")
    virtual_equity = 1000.0
    peak_equity = 1000.0
    max_drawdown = 0.0
    
    AI_STEP = 10 
    portfolio = {} 
    
    if os.path.exists("world_model.pt"): os.remove("world_model.pt")
    if os.path.exists("last_action.pt"): os.remove("last_action.pt")
        
    # We start deep enough to ensure macro rolling averages are stable
    for i in range(150, len(thermo_df)):
        
        step_pnl = 0.0
        current_phi_M1 = phi_M1[i].item()
        current_phi_H1 = phi_H1[i].item()
        ejected_positions = []
        
        # 1. True Unleveraged PnL Tracking
        for sym, pos in portfolio.items():
            current_price = thermo_df.iloc[i][f"{sym}_M1_close"]
            pct_return = (current_price - pos['entry_price']) / pos['entry_price']
            if pos['direction'] == 'SHORT':
                pct_return = -pct_return
            
            # PURE 1:1 YIELD. No artificial leverage. 
            trade_pnl = pos['size'] * pct_return 
            step_pnl += trade_pnl
            pos['size'] += trade_pnl # strict spot compounding
            
            # THE FRACTAL VOLCANIC GATE 
            # If EITHER the Micro (M1) or Macro (H1) manifold collapses into chaos, step off the board.
            if current_phi_M1 < 0.5 or current_phi_H1 < 0.5:
                ejected_positions.append(sym)
                
        for sym in ejected_positions:
            del portfolio[sym]
            
        virtual_equity += step_pnl
        
        if virtual_equity > peak_equity:
            peak_equity = virtual_equity
        dd = (peak_equity - virtual_equity) / peak_equity
        if dd > max_drawdown:
            max_drawdown = dd
            
        # Update rolling entry prices for compounded calculation
        for sym in portfolio:
            portfolio[sym]['entry_price'] = thermo_df.iloc[i][f"{sym}_M1_close"]
            
        # 2. AI Rebalancing Step 
        if i % AI_STEP == 0 and current_phi_M1 >= 0.5 and current_phi_H1 >= 0.5:
            portfolio.clear() 
            
            start_idx = max(0, i - 120)
            
            # --- THE RESONANCE INJECTION ---
            # We feed PyTorch the Micro M1 kinematics (the immediate trigger)...
            t_v, t_m = v_M1[start_idx:i+1], m_M1[start_idx:i+1]
            t_J, t_dJ, t_phi = J_M1[start_idx:i+1], dJ_M1[start_idx:i+1], phi_M1[start_idx:i+1]
            
            # ... BUT WE REPLACE M1 TENSION WITH H1 MACRO TENSION.
            # The AI is forced to optimize Free Energy by aligning the 1-minute Flux 
            # with the 1-hour coiled structural spring.
            t_q_MACRO = q_H1[start_idx:i+1]
            
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                final_actions, _ = po.decide_active_inference(active_nodes, active_edges, t_v, t_m, t_q_MACRO, t_J, t_dJ, t_phi)
            except Exception:
                final_actions = None
            finally:
                sys.stdout.close()
                sys.stdout = old_stdout
                
            if final_actions is not None:
                total_conviction = np.sum(np.abs(final_actions)) + 1e-6
                
                for idx, edge in enumerate(active_edges):
                    sym = edge['symbol']
                    u_risk = final_actions[idx]
                    
                    if abs(u_risk) > 0.05: 
                        weight = abs(u_risk) / total_conviction
                        # Absolute Risk Limit: Max 15% physical capital allocation
                        allocation = virtual_equity * 0.15 * weight 
                        
                        portfolio[sym] = {
                            'direction': 'LONG' if u_risk > 0 else 'SHORT',
                            'size': allocation,
                            'entry_price': thermo_df.iloc[i][f"{sym}_M1_close"]
                        }
                        
        if i % (AI_STEP * 20) == 0:
            print(f"  -> [\u03C4 State {i}/{len(thermo_df)}] True Equity: ${virtual_equity:.2f} | MDD: {max_drawdown*100:.2f}% | Active Pos: {len(portfolio)}")

    print("\n" + "="*50)
    print("FRACTAL 6-MONTH PURE ALPHA AUDIT (1:1 YIELD)")
    print("="*50)
    print(f"Final Equity:   ${virtual_equity:.2f} (Starting: $1000.00)")
    print(f"Total Yield:    {((virtual_equity - 1000)/1000)*100:.2f}% (UNLEVERAGED)")
    print(f"Max Drawdown:   {max_drawdown*100:.2f}% (UNLEVERAGED)")
    print("="*50)

if __name__ == "__main__":
    run_fractal_backtest()
