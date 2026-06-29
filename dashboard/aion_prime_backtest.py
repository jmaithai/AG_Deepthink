"""
AION PRIME HISTORICAL BACKTEST
-------------------------------
Feeds the corrected 222M tick parquet dataset through the AION Prime doctrine.
Global Event Clock: every GLOBAL_THRESHOLD ticks across all symbols = 1 state snapshot.
This is the original, pure AION doctrine on real data, with costs and sanity checks.
"""
import pandas as pd
import numpy as np
import glob
import os
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# ---- DOCTRINE CONFIG ----
GLOBAL_THRESHOLD = 25000   # Global volume event clock per AION Prime doctrine
DYNAMIC_WINDOW   = 50      # Rolling manifold memory
SIGMA_THRESHOLD  = 3.0     # 3-sigma coiled node detection

# ---- BACKTEST REALITY CONFIG ----
STARTING_BALANCE = 100000.0
FIXED_RISK       = 1000.0  # Fixed $ risk per trade to prevent compounding artifacts
TRANSACTION_COST = 5.0     # Estimated spread/commission cost in $ per trade
MAX_HOLD_STATES  = 100     # Force exit if stuck for 100 states (~100 * 25k ticks)

def load_data():
    files = sorted(glob.glob("tick_data/chunk_*.parquet"), reverse=True)
    dfs = []
    print(f"[*] Loading {len(files)} parquet chunks...")
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)
    full_df.sort_values(by='time_msc', inplace=True)
    full_df.reset_index(drop=True, inplace=True)
    full_df['datetime'] = pd.to_datetime(full_df['time_msc'], unit='ms')
    print(f"[+] Loaded {len(full_df):,} ticks across {full_df['symbol'].nunique()} symbols.")
    print(f"[+] Span: {full_df['datetime'].min()} to {full_df['datetime'].max()}")
    return full_df

def run_aion_prime(full_df):
    active_symbols = full_df['symbol'].unique().tolist()
    n = len(active_symbols)
    
    shadow_balance  = STARTING_BALANCE
    shadow_trades   = {}
    closed_trades   = []
    trade_id_counter = 0
    
    state_memory    = []
    current_prices  = {sym: [] for sym in active_symbols}
    latest_price    = {sym: 0.0 for sym in active_symbols}
    
    # GLOBAL event clock — counts every tick across all dimensions
    global_tick_count = 0
    state_idx = 0
    
    print(f"\n[*] AION PRIME DOCTRINE ENGAGED")
    print(f"[*] Global Event Clock Threshold: {GLOBAL_THRESHOLD:,} ticks")
    print(f"[*] Fixed Risk per Trade: ${FIXED_RISK}")
    print(f"[*] Transaction Cost: ${TRANSACTION_COST} per trade")
    print(f"[*] Max Hold States: {MAX_HOLD_STATES}")
    print(f"[*] Processing {len(full_df):,} ticks...\n")
    
    for i, row in enumerate(full_df.itertuples()):
        sym = row.symbol
        p   = row.price
        
        latest_price[sym] = p
        current_prices[sym].append(p)
        global_tick_count += 1
        
        # Global event clock fires
        if global_tick_count >= GLOBAL_THRESHOLD:
            global_tick_count = 0
            
            # Snapshot manifold state
            state_close = []
            for s in active_symbols:
                if current_prices[s]:
                    state_close.append(np.mean(current_prices[s]))
                else:
                    state_close.append(latest_price[s] if latest_price[s] > 0 else np.nan)
                    
            state_memory.append(state_close)
            if len(state_memory) > DYNAMIC_WINDOW + 10:
                state_memory.pop(0)
                
            # Reset accumulators
            current_prices = {sym: [] for sym in active_symbols}
            state_idx += 1
            
            if state_idx % 10 == 0:
                print(f"  [~] Manifold State {state_idx} | Open: {len(shadow_trades)} | Closed: {len(closed_trades)} | Balance: ${shadow_balance:,.2f}")
            
            if len(state_memory) < DYNAMIC_WINDOW:
                continue
                
            prices = np.array(state_memory[-DYNAMIC_WINDOW:], dtype=float)
            
            # Fill any NaN columns forward
            for col in range(prices.shape[1]):
                mask = np.isnan(prices[:, col])
                if mask.any():
                    prices[~mask, col] = prices[~mask, col]
                    idx_arr = np.arange(len(prices))
                    prices[:, col] = np.interp(idx_arr, idx_arr[~mask], prices[~mask, col]) if (~mask).sum() > 1 else prices[:, col]
                    
            # Build velocity field + graph Laplacian
            velocity     = np.diff(prices, axis=0, prepend=prices[0:1])
            std_v        = np.std(velocity, axis=0) + 1e-12
            v_norm       = velocity / std_v
            
            corr = np.corrcoef(v_norm.T)
            np.fill_diagonal(corr, 0)
            adj  = np.clip(corr, 0, None)
            
            deg       = np.diag(np.sum(adj, axis=1))
            laplacian = deg - adj
            current_v      = velocity[-1] / std_v
            network_force  = -np.dot(laplacian, current_v)
            
            # ---- EXITS ----
            for t_id in list(shadow_trades.keys()):
                trade  = shadow_trades[t_id]
                tsym   = trade['symbol']
                if tsym not in active_symbols:
                    continue
                tidx   = active_symbols.index(tsym)
                curr_p = prices[-1, tidx]
                
                direction   = trade['direction']
                stop_price  = trade['stop_price']
                barycenter  = trade['barycenter']
                risk_amount = trade['risk_amount']
                entry_state = trade['entry_state']
                
                closed = False
                pnl    = 0.0
                reason = ""
                
                # Force exit if held too long
                if state_idx - entry_state >= MAX_HOLD_STATES:
                    if direction == 'LONG':
                        dist   = curr_p - trade['entry_price']
                        sl_dist = abs(trade['entry_price'] - trade['stop_price'])
                    else:
                        dist   = trade['entry_price'] - curr_p
                        sl_dist = abs(trade['stop_price'] - trade['entry_price'])
                    pnl_raw = risk_amount * (dist / sl_dist) if sl_dist > 0 else 0
                    pnl = pnl_raw - TRANSACTION_COST
                    closed = True
                    reason = "MAX_HOLD_TIME"
                else:
                    if direction == 'LONG':
                        if curr_p <= stop_price:
                            pnl = -risk_amount - TRANSACTION_COST
                            closed = True
                            reason = "STOP_LOSS"
                        elif curr_p >= barycenter:
                            dist   = abs(barycenter - trade['entry_price'])
                            sl_dist = abs(trade['entry_price'] - stop_price)
                            pnl_raw = risk_amount * (dist / sl_dist) if sl_dist > 0 else 0
                            pnl = pnl_raw - TRANSACTION_COST
                            closed = True
                            reason = "TARGET_HIT"
                    else:
                        if curr_p >= stop_price:
                            pnl = -risk_amount - TRANSACTION_COST
                            closed = True
                            reason = "STOP_LOSS"
                        elif curr_p <= barycenter:
                            dist   = abs(trade['entry_price'] - barycenter)
                            sl_dist = abs(stop_price - trade['entry_price'])
                            pnl_raw = risk_amount * (dist / sl_dist) if sl_dist > 0 else 0
                            pnl = pnl_raw - TRANSACTION_COST
                            closed = True
                            reason = "TARGET_HIT"
                        
                if closed:
                    trade['pnl']       = pnl
                    trade['exit_reason'] = reason
                    shadow_balance    += pnl
                    closed_trades.append(trade)
                    del shadow_trades[t_id]
            
            # ---- ENTRIES: Alpha Field + KDE Topology ----
            alphas = np.zeros(n)
            kdes   = {}
            
            for idx in range(n):
                hist_p = prices[:, idx]
                p_min, p_max = np.min(hist_p), np.max(hist_p)
                if p_max == p_min:
                    continue
                try:
                    kde = gaussian_kde(hist_p, bw_method='silverman')
                    kdes[idx] = kde
                except:
                    continue
                    
                curr_p      = prices[-1, idx]
                local_dens  = kde(curr_p)[0]
                avg_dens    = np.mean(kde(np.linspace(p_min, p_max, 50))) + 1e-6
                push_up     = network_force[idx] > 0
                p_range     = p_max - p_min
                
                if push_up:
                    void_eval = np.linspace(curr_p, curr_p + p_range * 0.1, 10)
                else:
                    void_eval = np.linspace(curr_p - p_range * 0.1, curr_p, 10)
                    
                void_dens      = np.mean(kde(void_eval))
                norm_force     = abs(network_force[idx])
                alphas[idx]    = (norm_force * (local_dens / avg_dens)) / ((void_dens / avg_dens) + 1e-6)
                
            mean_alpha = np.mean(alphas)
            std_alpha  = np.std(alphas)  + 1e-6
            
            for idx in range(n):
                sigma_dev = (alphas[idx] - mean_alpha) / std_alpha
                if sigma_dev < SIGMA_THRESHOLD or alphas[idx] < 5.0:
                    continue
                    
                # Must be coiled (force opposing velocity or stalled)
                if not (np.sign(network_force[idx]) != np.sign(current_v[idx]) or abs(current_v[idx]) < 0.1):
                    continue
                    
                trap_sym  = active_symbols[idx]
                push_up   = network_force[idx] > 0
                hist_p    = prices[:, idx]
                p_min, p_max = np.min(hist_p), np.max(hist_p)
                p_range   = p_max - p_min
                curr_p    = prices[-1, idx]
                
                # Topographic stop via KDE density peak
                wall_price = curr_p
                if idx in kdes:
                    try:
                        kde    = kdes[idx]
                        eval_p = np.linspace(p_min, p_max, 100)
                        dens   = kde(eval_p)
                        peaks, _ = find_peaks(dens)
                        if len(peaks) > 0:
                            wall_price = eval_p[peaks[np.argmax(dens[peaks])]]
                    except:
                        pass
                        
                if push_up:
                    stop_dist = curr_p - (wall_price - p_range * 0.2)
                else:
                    stop_dist = (wall_price + p_range * 0.2) - curr_p
                stop_dist = max(stop_dist, 1e-5)
                
                # Ripple — highest correlated secondary node
                corr_row = adj[idx].copy()
                corr_row[idx] = -1
                secondary_idx = np.argmax(corr_row)
                ripple_sym    = active_symbols[secondary_idx]
                
                # FILTER: ONLY trade Metals and the Pound
                allowed = ['XAG', 'XAU', 'GAG', 'XPT', 'XPD', 'GBP']
                if not any(a in ripple_sym for a in allowed):
                    continue
                
                ripple_curr_p   = latest_price[ripple_sym]
                if ripple_curr_p <= 0:
                    continue
                    
                ripple_hist_p   = prices[:, secondary_idx]
                ripple_barycenter = np.mean(ripple_hist_p)
                vacuum_dist     = abs(ripple_curr_p - ripple_barycenter)
                friction        = ripple_curr_p * 0.0002
                
                if vacuum_dist <= friction:
                    continue
                    
                corr_weight = corr_row[secondary_idx]
                r_dir = "LONG" if (push_up and corr_weight > 0) or (not push_up and corr_weight < 0) else "SHORT"
                
                # FILTER: Do not enter if the trade direction is already exhausted past barycenter
                if r_dir == "LONG" and ripple_curr_p >= ripple_barycenter:
                    continue
                if r_dir == "SHORT" and ripple_curr_p <= ripple_barycenter:
                    continue
                    
                # No duplicate exposure
                if any(t['symbol'] == ripple_sym for t in shadow_trades.values()):
                    continue
                    
                # Correct stop on target symbol's own range
                ripple_p_max = np.max(ripple_hist_p)
                ripple_p_min = np.min(ripple_hist_p)
                ripple_p_range = max(ripple_p_max - ripple_p_min, ripple_curr_p * 0.001)
                
                if r_dir == "LONG":
                    r_stop = ripple_p_min - ripple_p_range * 0.2
                    r_stop = min(r_stop, ripple_curr_p - friction * 2)
                else:
                    r_stop = ripple_p_max + ripple_p_range * 0.2
                    r_stop = max(r_stop, ripple_curr_p + friction * 2)
                    
                risk_pct = min(5.0, max(1.0, 1.0 + (sigma_dev - SIGMA_THRESHOLD) * 2.0))
                risk_amt = shadow_balance * (risk_pct / 100.0)
                trade_id_counter += 1
                
                shadow_trades[trade_id_counter] = {
                    'id':          trade_id_counter,
                    'symbol':      ripple_sym,
                    'trigger':     trap_sym,
                    'direction':   r_dir,
                    'entry_price': ripple_curr_p,
                    'stop_price':  r_stop,
                    'barycenter':  ripple_barycenter,
                    'risk_amount': risk_amt,
                    'sigma':       sigma_dev,
                    'entry_state': state_idx,
                }
                
    # Close all open trades at last known prices
    for t_id in list(shadow_trades.keys()):
        trade  = shadow_trades[t_id]
        tsym   = trade['symbol']
        curr_p = latest_price[tsym]
        if trade['direction'] == 'LONG':
            dist   = curr_p - trade['entry_price']
            sl_dist = abs(trade['entry_price'] - trade['stop_price'])
        else:
            dist   = trade['entry_price'] - curr_p
            sl_dist = abs(trade['stop_price'] - trade['entry_price'])
        pnl_raw = trade['risk_amount'] * (dist / sl_dist) if sl_dist > 0 else 0
        pnl = pnl_raw - TRANSACTION_COST
        trade['pnl'] = pnl
        trade['exit_reason'] = "END_OF_BACKTEST"
        shadow_balance += pnl
        closed_trades.append(trade)
        
    return closed_trades, shadow_balance

def print_results(closed_trades, final_balance):
    print("\n" + "="*70)
    print("AION PRIME HISTORICAL BACKTEST — FINAL RESULTS")
    print("="*70)
    total  = len(closed_trades)
    if total == 0:
        print("[-] No trades generated. Reduce GLOBAL_THRESHOLD or check data.")
        return
    wins   = [t for t in closed_trades if t['pnl'] > 0]
    losses = [t for t in closed_trades if t['pnl'] <= 0]
    gross_profit = sum(t['pnl'] for t in wins)
    gross_loss   = abs(sum(t['pnl'] for t in losses))
    pf   = gross_profit / gross_loss if gross_loss > 0 else 999.9
    wr   = len(wins) / total * 100
    
    print(f"Starting Balance : ${STARTING_BALANCE:>12,.2f}")
    print(f"Final Balance    : ${final_balance:>12,.2f}")
    print(f"Net PnL          : ${final_balance - STARTING_BALANCE:>+12,.2f}  ({(final_balance/STARTING_BALANCE - 1)*100:+.2f}%)")
    print(f"Total Trades     : {total}")
    print(f"Win Rate         : {wr:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"Profit Factor    : {pf:.2f}")
    print(f"Avg Win          : ${gross_profit/len(wins):,.2f}" if wins else "Avg Win: N/A")
    print(f"Avg Loss         : ${gross_loss/len(losses):,.2f}" if losses else "Avg Loss: N/A")
    
    # Analyze Exit Reasons
    from collections import Counter
    reasons = Counter(t.get('exit_reason', 'UNKNOWN') for t in closed_trades)
    print(f"\nExit Reasons:")
    for r, c in reasons.items():
        print(f"  {r:<15}: {c}")

    # Top pairs
    sym_pnl = {}
    for t in closed_trades:
        sym_pnl.setdefault(t['symbol'], []).append(t['pnl'])
    print(f"\nTop 5 symbols by PnL:")
    ranked = sorted(sym_pnl.items(), key=lambda x: sum(x[1]), reverse=True)[:5]
    for sym, pnls in ranked:
        print(f"  {sym:<12} Trades: {len(pnls):>3} | PnL: ${sum(pnls):>+9,.2f} | WR: {sum(1 for p in pnls if p>0)/len(pnls)*100:.0f}%")
    print("="*70)

if __name__ == "__main__":
    full_df = load_data()
    closed_trades, final_balance = run_aion_prime(full_df)
    print_results(closed_trades, final_balance)
