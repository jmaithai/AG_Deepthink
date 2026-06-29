import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import time
import os
import json
import optuna

def objective(trial, df, active_symbols):
    # Optuna suggests a threshold for each symbol
    thresholds = {}
    for sym in active_symbols:
        thresholds[sym] = trial.suggest_int(f"th_{sym}", 100, 5000)
        
    shadow_balance = 100000.0
    shadow_trades = {}
    closed_trades = []
    trade_id_counter = 0
    
    dynamic_window = 20 # Smaller window for faster evaluation during optimization
    
    state_memory = []
    current_prices = {sym: [] for sym in active_symbols}
    cumulative_ticks = {sym: 0 for sym in active_symbols}
    latest_price = {sym: 0.0 for sym in active_symbols}
    
    # Process ticks
    for row in df.itertuples():
        sym = row.symbol
        p = row.price
        latest_price[sym] = p
        
        current_prices[sym].append(p)
        cumulative_ticks[sym] += 1
        
        # When ANY symbol breaches its personal threshold, trigger global snapshot
        if cumulative_ticks[sym] >= thresholds[sym]:
            # Reset this symbol's clock
            cumulative_ticks[sym] = 0
            
            # Take a global snapshot of all nodes
            state_close = []
            for s in active_symbols:
                if current_prices[s]:
                    state_close.append(np.mean(current_prices[s]))
                else:
                    state_close.append(latest_price[s] if latest_price[s] > 0 else np.nan)
                    
            state_memory.append(state_close)
            
            if len(state_memory) > dynamic_window + 50:
                state_memory.pop(0)
                
            if len(state_memory) >= dynamic_window:
                df_mem = pd.DataFrame(state_memory[-dynamic_window:], columns=active_symbols)
                df_mem = df_mem.ffill().bfill()
                prices = df_mem.values
                
                velocity = np.diff(prices, axis=0, prepend=prices[0:1])
                std_v = np.std(velocity, axis=0) + 1e-12
                v_norm = velocity / std_v
                
                corr = np.corrcoef(v_norm.T)
                np.fill_diagonal(corr, 0)
                adj = np.clip(corr, 0, None)
                
                deg = np.diag(np.sum(adj, axis=1))
                laplacian = deg - adj
                
                current_v = velocity[-1] / std_v
                network_force = -np.dot(laplacian, current_v)
                
                # SHADOW TRADING EVALUATION (Fast Vectorized Approximation)
                # Instead of full KDE evaluation for speed, we evaluate open trades
                for t_id in list(shadow_trades.keys()):
                    trade = shadow_trades[t_id]
                    tsym = trade['symbol']
                    if tsym in active_symbols:
                        idx = active_symbols.index(tsym)
                        curr_p = prices[-1, idx]
                        
                        direction = trade['direction']
                        stop_price = trade['stop_price']
                        barycenter = trade['barycenter']
                        risk_amount = trade['risk_amount']
                        
                        closed = False
                        pnl = 0.0
                        
                        if direction == 'LONG':
                            if curr_p <= stop_price:
                                pnl = -risk_amount
                                closed = True
                            elif curr_p >= barycenter:
                                dist = abs(barycenter - trade['entry_price'])
                                sl_dist = abs(trade['entry_price'] - stop_price)
                                pnl = risk_amount * (dist / sl_dist) if sl_dist > 0 else 0
                                closed = True
                        else:
                            if curr_p >= stop_price:
                                pnl = -risk_amount
                                closed = True
                            elif curr_p <= barycenter:
                                dist = abs(trade['entry_price'] - barycenter)
                                sl_dist = abs(stop_price - trade['entry_price'])
                                pnl = risk_amount * (dist / sl_dist) if sl_dist > 0 else 0
                                closed = True
                                
                        if closed:
                            trade['pnl'] = pnl
                            shadow_balance += pnl
                            closed_trades.append(trade)
                            del shadow_trades[t_id]
                
                # Calculate Sigma (Approximation for Optimizer Speed)
                norm_force = np.abs(network_force)
                mean_force = np.mean(norm_force) + 1e-6
                std_force = np.std(norm_force) + 1e-6
                sigma_devs = (norm_force - mean_force) / std_force
                
                for idx in range(len(active_symbols)):
                    if sigma_devs[idx] > 2.5: # Lowered threshold slightly for OOS hits
                        if np.sign(network_force[idx]) != np.sign(current_v[idx]) or abs(current_v[idx]) < 0.1:
                            push_up = network_force[idx] > 0
                            direction = "LONG" if push_up else "SHORT"
                            
                            curr_p = prices[-1, idx]
                            
                            # Simple wall approximation for optimizer
                            p_max = np.max(prices[:, idx])
                            p_min = np.min(prices[:, idx])
                            p_range = p_max - p_min
                            if p_range == 0: continue
                            
                            if push_up:
                                stop_dist = (curr_p - (p_min - p_range * 0.2))
                            else:
                                stop_dist = ((p_max + p_range * 0.2) - curr_p)
                            stop_dist = max(stop_dist, 0.001)
                            
                            risk_pct = min(5.0, max(1.0, 1.0 + (sigma_devs[idx] - 2.5) * 2.0))
                            
                            corr_row = adj[idx].copy()
                            corr_row[idx] = -1 
                            secondary_idx = np.argmax(corr_row)
                            ripple_sym = active_symbols[secondary_idx]
                            
                            ripple_curr_p = latest_price[ripple_sym]
                            if ripple_curr_p > 0:
                                ripple_barycenter = np.mean(prices[:, secondary_idx])
                                vacuum_dist = abs(ripple_curr_p - ripple_barycenter)
                                friction = ripple_curr_p * 0.0002
                                
                                if vacuum_dist > friction:
                                    # Prevent duplicate exposure
                                    already_active = False
                                    for t_val in shadow_trades.values():
                                        if t_val['symbol'] == ripple_sym:
                                            already_active = True
                                            break
                                    if already_active:
                                        continue

                                    risk_amt = shadow_balance * (risk_pct / 100.0)
                                    trade_id_counter += 1
                                    
                                    ripple_hist_p = prices[:, secondary_idx]
                                    ripple_p_max = np.max(ripple_hist_p)
                                    ripple_p_min = np.min(ripple_hist_p)
                                    ripple_p_range = ripple_p_max - ripple_p_min
                                    if ripple_p_range == 0:
                                        ripple_p_range = ripple_curr_p * 0.001

                                    corr_weight = corr_row[secondary_idx]
                                    r_dir = "LONG" if (push_up and corr_weight > 0) or (not push_up and corr_weight < 0) else "SHORT"
                                    
                                    if r_dir == "LONG":
                                        r_stop = ripple_p_min - (ripple_p_range * 0.2)
                                        if ripple_curr_p - r_stop < friction * 2:
                                            r_stop = ripple_curr_p - friction * 2
                                    else:
                                        r_stop = ripple_p_max + (ripple_p_range * 0.2)
                                        if r_stop - ripple_curr_p < friction * 2:
                                            r_stop = ripple_curr_p + friction * 2
                                    
                                    shadow_trades[trade_id_counter] = {
                                        'id': trade_id_counter,
                                        'symbol': ripple_sym,
                                        'direction': r_dir,
                                        'entry_price': ripple_curr_p,
                                        'stop_price': r_stop,
                                        'barycenter': ripple_barycenter,
                                        'risk_amount': risk_amt
                                    }
                                    
    total_trades = len(closed_trades)
    if total_trades < 5:
        return -100.0 # Punish the optimizer for not trading
        
    wins = sum([1 for t in closed_trades if t['pnl'] > 0])
    gross_profit = sum([t['pnl'] for t in closed_trades if t['pnl'] > 0])
    gross_loss = abs(sum([t['pnl'] for t in closed_trades if t['pnl'] < 0]))
    
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 5.0
    win_rate = wins / total_trades
    
    # We want to maximize Profit Factor * Win Rate (Sharpe proxy)
    score = profit_factor * (win_rate * 100)
    
    # Penalize if it only traded a few times
    if total_trades < 10:
        score *= (total_trades / 10.0)
        
    return score

def main():
    if not os.path.exists("ticks_history.pkl"):
        print("ticks_history.pkl not found!")
        return
        
    print("Loading historical ticks for 38-D Optimizer...")
    df = pd.read_pickle("ticks_history.pkl")
    
    # Take a 100,000 tick slice for fast optimization testing
    print("Taking 100,000 tick subset for optimization...")
    df_subset = df.iloc[0:100000].copy()
    active_symbols = df_subset['symbol'].unique().tolist()
    print(f"Active dimensions: {len(active_symbols)}")
    
    print("\n[*] Initializing Optuna Bayesian Optimizer...")
    study = optuna.create_study(direction="maximize")
    
    start_time = time.time()
    # Run 10 trials as a proof of concept
    study.optimize(lambda trial: objective(trial, df_subset, active_symbols), n_trials=10)
    
    print(f"\nOptimization completed in {time.time() - start_time:.2f}s")
    
    print("\n--- 38-DIMENSIONAL GLOBAL OPTIMUM FOUND ---")
    best_params = study.best_params
    best_score = study.best_value
    print(f"Optimal Sharpe Score: {best_score:.2f}")
    
    # Save the tensor
    optimal_tensor = {k.replace('th_', ''): v for k, v in best_params.items()}
    with open("optimal_tensor.json", "w") as f:
        json.dump(optimal_tensor, f, indent=4)
    print("\n[+] Exported optimal_tensor.json")
    
    print("\nOptimal 38-D Threshold Tensor:")
    
    # Group output nicely
    items = list(best_params.items())
    for i in range(0, len(items), 4):
        chunk = items[i:i+4]
        line = " | ".join([f"{k.replace('th_', '')}: {v}" for k, v in chunk])
        print(line)

if __name__ == "__main__":
    main()
