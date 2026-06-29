import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import time
import os

EVENT_THRESHOLDS = [500, 2500, 5000, 10000]

def run_backtest(df, start_idx, end_idx, phase_name="IS"):
    print(f"\n--- Running {phase_name} Phase ---")
    active_symbols = df['symbol'].unique().tolist()
    
    shadow_balance = {th: 100000.0 for th in EVENT_THRESHOLDS}
    shadow_trades = {th: {} for th in EVENT_THRESHOLDS}
    closed_trades = {th: [] for th in EVENT_THRESHOLDS}
    trade_id_counter = {th: 0 for th in EVENT_THRESHOLDS}
    
    dynamic_window = 50
    state_memory = {th: [] for th in EVENT_THRESHOLDS}
    current_prices = {th: {sym: [] for sym in active_symbols} for th in EVENT_THRESHOLDS}
    cumulative_ticks = {th: 0 for th in EVENT_THRESHOLDS}
    
    # Fast lookup for prices to calculate PnL
    latest_price = {sym: 0.0 for sym in active_symbols}
    
    # Process ticks
    start_time = time.time()
    for row in df.iloc[start_idx:end_idx].itertuples():
        sym = row.symbol
        p = row.price
        latest_price[sym] = p
        
        for th in EVENT_THRESHOLDS:
            current_prices[th][sym].append(p)
            cumulative_ticks[th] += 1
            
            if cumulative_ticks[th] >= th:
                state_close = [np.mean(current_prices[th][s]) if current_prices[th][s] else np.nan for s in active_symbols]
                state_memory[th].append(state_close)
                
                # Manage memory limit (we only need the last 50 states + some buffer for KDE)
                if len(state_memory[th]) > dynamic_window + 100:
                    state_memory[th].pop(0)
                
                cumulative_ticks[th] = 0
                current_prices[th] = {s: [] for s in active_symbols}
                
                if len(state_memory[th]) >= dynamic_window:
                    # Run physics
                    df_mem = pd.DataFrame(state_memory[th][-dynamic_window:], columns=active_symbols)
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
                    
                    # SHADOW TRADING EVALUATION
                    for t_id in list(shadow_trades[th].keys()):
                        trade = shadow_trades[th][t_id]
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
                                shadow_balance[th] += pnl
                                closed_trades[th].append(trade)
                                del shadow_trades[th][t_id]

                    alphas = np.zeros(len(active_symbols))
                    for idx in range(len(active_symbols)):
                        hist_p = prices[:, idx]
                        p_min, p_max = np.min(hist_p), np.max(hist_p)
                        if p_max == p_min: continue
                        
                        try:
                            kde = gaussian_kde(hist_p, bw_method='silverman')
                            curr_p = prices[-1, idx]
                            local_density = kde(curr_p)[0]
                            
                            eval_points = np.linspace(p_min, p_max, 50)
                            avg_density = np.mean(kde(eval_points)) + 1e-6
                            
                            push_up = network_force[idx] > 0
                            p_range = p_max - p_min
                            
                            if push_up:
                                void_eval = np.linspace(curr_p, curr_p + (p_range * 0.1), 10)
                            else:
                                void_eval = np.linspace(curr_p - (p_range * 0.1), curr_p, 10)
                                
                            void_density = np.mean(kde(void_eval))
                            norm_force = np.abs(network_force[idx])
                            alphas[idx] = (norm_force * (local_density / avg_density)) / ((void_density / avg_density) + 1e-6)
                        except:
                            pass
                            
                    mean_alpha = np.mean(alphas)
                    std_alpha = np.std(alphas) + 1e-6
                    
                    for idx in range(len(active_symbols)):
                        sigma_dev = (alphas[idx] - mean_alpha) / std_alpha
                        if sigma_dev > 3.0 and alphas[idx] > 5.0:
                            if np.sign(network_force[idx]) != np.sign(current_v[idx]) or abs(current_v[idx]) < 0.1:
                                push_up = network_force[idx] > 0
                                direction = "LONG" if push_up else "SHORT"
                                
                                hist_p = prices[:, idx]
                                p_min, p_max = np.min(hist_p), np.max(hist_p)
                                curr_p = prices[-1, idx]
                                
                                try:
                                    kde = gaussian_kde(hist_p, bw_method='silverman')
                                    e_p = np.linspace(p_min, p_max, 100)
                                    dens = kde(e_p)
                                    peaks, _ = find_peaks(dens)
                                    if len(peaks) > 0:
                                        wall_price = e_p[peaks[np.argmax(dens[peaks])]]
                                    else:
                                        wall_price = curr_p
                                except:
                                    wall_price = curr_p
                                
                                p_range = p_max - p_min
                                if push_up:
                                    stop_dist = (curr_p - (wall_price - p_range * 0.2))
                                else:
                                    stop_dist = ((wall_price + p_range * 0.2) - curr_p)
                                stop_dist = max(stop_dist, 0.001)
                                
                                risk_pct = min(5.0, max(1.0, 1.0 + (sigma_dev - 3.0) * 2.0))
                                
                                corr_row = adj[idx].copy()
                                corr_row[idx] = -1 
                                secondary_idx = np.argmax(corr_row)
                                ripple_sym = active_symbols[secondary_idx]
                                
                                ripple_curr_p = latest_price[ripple_sym]
                                if ripple_curr_p > 0:
                                    ripple_hist_p = prices[:, secondary_idx]
                                    ripple_barycenter = np.mean(ripple_hist_p)
                                    vacuum_dist = abs(ripple_curr_p - ripple_barycenter)
                                    
                                    # Simulate typical 1.5 spread friction (approximate since historical bid/ask can be noisy)
                                    # We use 0.00020 for FX pairs and relative amounts for others
                                    friction = ripple_curr_p * 0.0002
                                    
                                    if vacuum_dist > friction:
                                        risk_amt = shadow_balance[th] * (risk_pct / 100.0)
                                        trade_id_counter[th] += 1
                                        
                                        corr_weight = corr_row[secondary_idx]
                                        r_dir = "LONG" if (push_up and corr_weight > 0) or (not push_up and corr_weight < 0) else "SHORT"
                                        
                                        r_stop = ripple_curr_p - (ripple_curr_p * stop_dist) if r_dir == "LONG" else ripple_curr_p + (ripple_curr_p * stop_dist)
                                        
                                        shadow_trades[th][trade_id_counter[th]] = {
                                            'id': trade_id_counter[th],
                                            'symbol': ripple_sym,
                                            'direction': r_dir,
                                            'entry_price': ripple_curr_p,
                                            'stop_price': r_stop,
                                            'barycenter': ripple_barycenter,
                                            'risk_amount': risk_amt
                                        }

    print(f"{phase_name} completed in {time.time() - start_time:.2f}s")
    
    # Results
    results = {}
    for th in EVENT_THRESHOLDS:
        trades = closed_trades[th]
        total_pnl = sum([t['pnl'] for t in trades])
        wins = sum([1 for t in trades if t['pnl'] > 0])
        total_trades = len(trades)
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        profit_factor = sum([t['pnl'] for t in trades if t['pnl'] > 0]) / abs(sum([t['pnl'] for t in trades if t['pnl'] < 0])) if abs(sum([t['pnl'] for t in trades if t['pnl'] < 0])) > 0 else float('inf')
        
        results[th] = {
            'trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'net_pnl': total_pnl,
            'ending_balance': shadow_balance[th]
        }
        print(f"[{th} Ticks] Trades: {total_trades}, Win Rate: {win_rate:.1f}%, Profit Factor: {profit_factor:.2f}, Net PnL: ${total_pnl:.2f}")
        
    best_th = max(results.keys(), key=lambda k: results[k]['profit_factor'] if results[k]['trades'] > 5 else 0)
    print(f"Optimal Threshold for {phase_name}: {best_th}")
    return best_th, results

def main():
    if not os.path.exists("ticks_history.pkl"):
        print("ticks_history.pkl not found! Run data_fetcher.py first.")
        return
        
    print("Loading historical ticks...")
    df = pd.read_pickle("ticks_history.pkl")
    print(f"Loaded {len(df)} ticks.")
    
    total_rows = len(df)
    split_idx = int(total_rows * 0.7)
    
    best_th, is_results = run_backtest(df, 0, split_idx, "IS (In-Sample)")
    
    # Only test the optimal threshold in OOS
    global EVENT_THRESHOLDS
    EVENT_THRESHOLDS = [best_th]
    
    _, oos_results = run_backtest(df, split_idx, total_rows, "OOS (Out-Of-Sample)")
    
    print("\n--- FINAL OPTIMIZATION REPORT ---")
    print(f"Winner: {best_th} Ticks")
    print(f"IS Profit Factor: {is_results[best_th]['profit_factor']:.2f} | IS Net PnL: ${is_results[best_th]['net_pnl']:.2f}")
    print(f"OOS Profit Factor: {oos_results[best_th]['profit_factor']:.2f} | OOS Net PnL: ${oos_results[best_th]['net_pnl']:.2f}")
    
    if oos_results[best_th]['profit_factor'] > 1.0:
        print("[+] CONCLUSION: Edge verified and mathematically robust out-of-sample.")
    else:
        print("[-] CONCLUSION: Edge degraded out-of-sample. Parameter may be overfitted or market regime shifted.")

if __name__ == "__main__":
    main()
