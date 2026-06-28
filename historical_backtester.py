import argparse
import pandas as pd
import numpy as np
import physics_engine as pe
import kinematics_engine as ke
import policy_engine as po
import execution_engine as ee
import csv

def run_simulation(parquet_file, dry_run=False):
    print(f"[*] Initializing Time Machine with: {parquet_file}")
    
    try:
        df = pd.read_parquet(parquet_file)
    except FileNotFoundError:
        print(f"[!] Critical Error: {parquet_file} not found. Run data_loader.py first.")
        return

    # 1. Batch Process Physics (Warp Time & Format)
    print(f"[*] Compressing {len(df)} Chronological M1 Bars into Proper Time...")
    nodes, edges = pe.discover_topology(data_source=df)
    thermo_df = pe.fetch_and_warp_fluid(edges, data_source=df)
    
    # 2. Extract Gauge Pressures
    print("[*] Extracting Hoberman Gauge Pressures...")
    pressure_df = pe.extract_hoberman_space(thermo_df, nodes, edges)
    
    # 3. Compute PyTorch Kinematics Tensors
    print("[*] Computing PyTorch Kinematics Tensors (Batch Vectorized)...")
    v, m, q, J, dJ, S_norm, phi = ke.compute_kinematics(pressure_df)
    
    # Simulation State
    virtual_equity = 1000.0
    print(f"\n[*] Starting Walk-Forward Simulation across {len(thermo_df)} Proper Time states. Equity: ${virtual_equity:.2f}")
    
    # Portfolio Memory
    previous_portfolio = {}
    all_logs = []
    
    edge_names = [e['symbol'] for e in edges]

    # 4. Time Machine Loop (Stepping through Proper Time)
    for i in range(21, len(thermo_df)):
        current_phi = phi[i].item()
        
        # Sliced history representing time exactly up to step 'i'
        _v = v[:i+1]
        _m = m[:i+1]
        _q = q[:i+1]
        _J = J[:i+1]
        _dJ = dJ[:i+1]
        _phi = phi[:i+1]
        
        # Inference (The Brain) using the sliced sequence
        final_actions, _ = po.decide_active_inference(nodes, edges, _v, _m, _q, _J, _dJ, _phi)
        
        # Virtual Execution
        if final_actions is not None and not dry_run:
            action_df = pd.DataFrame({'Symbol': edge_names, 'U_risk': final_actions})
            action_df['Abs_U'] = action_df['U_risk'].abs()
            action_df = action_df.sort_values(by='Abs_U', ascending=False)
            
            # Construct current_prices and prev_prices dictionary
            current_prices = {}
            for sym in edge_names:
                price_col = f"{sym}_price"
                if price_col in thermo_df.columns:
                    # Prices in thermo_df are log_prices! Need to exp them to get real prices
                    current_prices[sym] = np.exp(thermo_df.iloc[i][price_col])
                    current_prices['prev_' + sym] = np.exp(thermo_df.iloc[i-1][price_col])
            
            # Pass the instantaneous variables needed for logging
            current_S_norm = S_norm[i].numpy()
            current_dJ = dJ[i].numpy()
            
            step_pnl, new_portfolio, logs = ee.simulate_virtual_trade(
                action_df, 
                previous_portfolio, 
                current_prices, 
                virtual_equity, 
                current_phi, 
                current_S_norm, 
                current_dJ
            )
            
            virtual_equity += step_pnl
            previous_portfolio = new_portfolio
            all_logs.extend(logs)
            
        if i % 100 == 0:
            print(f"[Proper Time {i}/{len(thermo_df)}] Phi: {current_phi:.4f} | Equity: ${virtual_equity:.2f}")

    print(f"[+] Simulation Complete. Final Equity: ${virtual_equity:.2f}")
    
    if not dry_run and all_logs:
        log_df = pd.DataFrame(all_logs)
        log_df.to_csv("backtest.log", index=False)
        print(f"[+] Saved {len(log_df)} trade evaluations to backtest.log")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dreadnought Time Machine")
    parser.add_argument("--file", default="data_XAUUSD_M1.parquet", help="Path to Parquet data")
    parser.add_argument("--dry-run", action="store_true", help="Run without executing virtual trades")
    args = parser.parse_args()
    
    run_simulation(args.file, args.dry_run)
