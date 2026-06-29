import pandas as pd
import numpy as np
import os
import glob
import optuna
import time
from datetime import datetime, timedelta
import collections
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

from optimizer_38d import objective

def load_all_data():
    files = sorted(glob.glob("tick_data/chunk_*.parquet"), reverse=True)
    dfs = []
    print("[*] Loading all 1-month parquet chunks into memory (this may take a minute)...")
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)
    full_df.sort_values(by='time_msc', inplace=True)
    full_df.reset_index(drop=True, inplace=True)
    
    # Convert time_msc to datetime
    full_df['datetime'] = pd.to_datetime(full_df['time_msc'], unit='ms')
    return full_df

def run_oos_simulation(df_oos, thresholds, active_symbols, shadow_balance):
    # This matches the physics engine logic but runs purely Out-Of-Sample
    shadow_trades = {}
    closed_trades = []
    trade_id_counter = 0
    
    dynamic_window = 20 # Fast evaluation
    state_memory = []
    current_prices = {sym: collections.deque(maxlen=100) for sym in active_symbols}
    cumulative_ticks = {sym: 0 for sym in active_symbols}
    latest_price = {sym: 0.0 for sym in active_symbols}
    
    for row in df_oos.itertuples():
        sym = row.symbol
        p = row.price
        latest_price[sym] = p
        
        current_prices[sym].append(p)
        cumulative_ticks[sym] += 1
        
        threshold = thresholds.get(sym, 2500)
        
        if cumulative_ticks[sym] >= threshold:
            cumulative_ticks[sym] = 0
            
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
                
                # Check Exits
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
                
                # Check Entries
                norm_force = np.abs(network_force)
                mean_force = np.mean(norm_force) + 1e-6
                std_force = np.std(norm_force) + 1e-6
                sigma_devs = (norm_force - mean_force) / std_force
                
                for idx in range(len(active_symbols)):
                    if sigma_devs[idx] > 2.0:  # Lowered from 3.0 for WFO coverage
                        if np.sign(network_force[idx]) != np.sign(current_v[idx]) or abs(current_v[idx]) < 0.1:
                            push_up = network_force[idx] > 0
                            curr_p = prices[-1, idx]
                            
                            p_max = np.max(prices[:, idx])
                            p_min = np.min(prices[:, idx])
                            p_range = p_max - p_min
                            if p_range == 0: continue
                            
                            if push_up:
                                stop_dist = (curr_p - (p_min - p_range * 0.2))
                            else:
                                stop_dist = ((p_max + p_range * 0.2) - curr_p)
                            stop_dist = max(stop_dist, 0.001)
                            
                            risk_pct = min(5.0, max(1.0, 1.0 + (sigma_devs[idx] - 3.0) * 2.0))
                            
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
                                    
    # Close out any remaining trades at end of OOS period at current prices
    for t_id in list(shadow_trades.keys()):
        trade = shadow_trades[t_id]
        tsym = trade['symbol']
        curr_p = latest_price[tsym]
        
        if trade['direction'] == 'LONG':
            dist = curr_p - trade['entry_price']
            sl_dist = abs(trade['entry_price'] - trade['stop_price'])
            pnl = trade['risk_amount'] * (dist / sl_dist) if sl_dist > 0 else 0
        else:
            dist = trade['entry_price'] - curr_p
            sl_dist = abs(trade['stop_price'] - trade['entry_price'])
            pnl = trade['risk_amount'] * (dist / sl_dist) if sl_dist > 0 else 0
            
        trade['pnl'] = pnl
        shadow_balance += pnl
        closed_trades.append(trade)
        
    return closed_trades, shadow_balance

def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    full_df = load_all_data()
    active_symbols = full_df['symbol'].unique().tolist()
    
    start_time = full_df['datetime'].min()
    end_time = full_df['datetime'].max()
    
    print(f"\n[*] Data span: {start_time} to {end_time}")
    print(f"[*] Total Ticks: {len(full_df)}")
    
    IS_DAYS = 3
    OOS_DAYS = 1
    
    current_is_start = start_time
    global_balance = 100000.0
    all_oos_trades = []
    
    step_num = 1
    while True:
        current_is_end = current_is_start + timedelta(days=IS_DAYS)
        current_oos_end = current_is_end + timedelta(days=OOS_DAYS)
        
        if current_oos_end > end_time:
            break
            
        print(f"\n--- WFO Step {step_num} ---")
        print(f"[IS ] {current_is_start.strftime('%m-%d')} to {current_is_end.strftime('%m-%d')}")
        print(f"[OOS] {current_is_end.strftime('%m-%d')} to {current_oos_end.strftime('%m-%d')}")
        
        df_is = full_df[(full_df['datetime'] >= current_is_start) & (full_df['datetime'] < current_is_end)]
        df_oos = full_df[(full_df['datetime'] >= current_is_end) & (full_df['datetime'] < current_oos_end)]
        
        if len(df_is) < 10000 or len(df_oos) < 1000:
            print("[-] Insufficient data for this window. Skipping...")
            current_is_start += timedelta(days=OOS_DAYS)
            continue
            
        # Due to 200M rows, sub-sample IS to 100,000 ticks for fast optimization
        if len(df_is) > 100000:
            df_is_sub = df_is.sample(n=100000).sort_values(by='time_msc')
        else:
            df_is_sub = df_is
            
        print("[*] Running 38-D Optuna continuous optimization...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, df_is_sub, active_symbols), n_trials=5)
        
        best_params = study.best_params
        thresholds = {k.replace('th_', ''): v for k, v in best_params.items()}
        
        print("[*] Optimization complete. Running OOS Simulation...")
        oos_trades, new_balance = run_oos_simulation(df_oos, thresholds, active_symbols, global_balance)
        
        pnl = new_balance - global_balance
        wins = sum([1 for t in oos_trades if t['pnl'] > 0])
        total = len(oos_trades)
        win_rate = (wins / total * 100) if total > 0 else 0
        
        print(f"[+] OOS Results -> Trades: {total} | Win Rate: {win_rate:.1f}% | PnL: ${pnl:.2f}")
        print(f"[+] Global Balance: ${new_balance:.2f}")
        
        global_balance = new_balance
        all_oos_trades.extend(oos_trades)
        
        # Walk Forward
        current_is_start += timedelta(days=OOS_DAYS)
        step_num += 1
        
    print("\n--- FINAL 1-MONTH WFO METRICS ---")
    print(f"Total OOS Trades : {len(all_oos_trades)}")
    print(f"Starting Balance : $100000.00")
    print(f"Final Balance    : ${global_balance:.2f}")
    if len(all_oos_trades) > 0:
        total_wins = sum([1 for t in all_oos_trades if t['pnl'] > 0])
        total_gross_profit = sum([t['pnl'] for t in all_oos_trades if t['pnl'] > 0])
        total_gross_loss = abs(sum([t['pnl'] for t in all_oos_trades if t['pnl'] < 0]))
        pf = total_gross_profit / total_gross_loss if total_gross_loss > 0 else 999.9
        print(f"Global Win Rate  : {(total_wins/len(all_oos_trades))*100:.1f}%")
        print(f"Profit Factor    : {pf:.2f}")

if __name__ == "__main__":
    main()
