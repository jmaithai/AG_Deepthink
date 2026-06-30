import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import os
import warnings
from datetime import datetime
import argparse
import json
import collections
import threading
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# The 38-Dimensional Optimal Kinetic Tensor
TENSOR_FILE = "optimal_tensor.json"
OPTIMAL_THRESHOLDS = {}
last_tensor_mtime = 0

def load_tensor():
    global OPTIMAL_THRESHOLDS, last_tensor_mtime
    if os.path.exists(TENSOR_FILE):
        try:
            mtime = os.path.getmtime(TENSOR_FILE)
            if mtime > last_tensor_mtime:
                with open(TENSOR_FILE, 'r') as f:
                    new_tensor = json.load(f)
                    OPTIMAL_THRESHOLDS.update(new_tensor)
                last_tensor_mtime = mtime
                print(f"[*] Hot-reloaded Continuous Physics Tensor from {TENSOR_FILE}")
        except Exception as e:
            pass

def watch_tensor():
    while True:
        load_tensor()
        time.sleep(5)

# ---------------------------------------------------------
# AION DOCTRINE CONFIGURATION: LIVE SINGULARITY SENTINEL
# ---------------------------------------------------------

EVENT_THRESHOLD = 25000  # Restored default from backtest 
DRY_RUN = True           
MAX_FARM_MASS = 50  
TERMINAL_PATH = None
BROKER_NAME = "DEFAULT"
MAX_DRAWDOWN = 15.0
MAX_CONCURRENT_TRADES = 5

active_farms = {} 

def load_config():
    global EVENT_THRESHOLD, DRY_RUN, MAX_FARM_MASS, TERMINAL_PATH, BROKER_NAME, MAX_DRAWDOWN, MAX_CONCURRENT_TRADES
    parser = argparse.ArgumentParser(description='AION Live Singularity Sentinel')
    parser.add_argument('--config', type=str, help='Path to broker config JSON')
    
    args, unknown = parser.parse_known_args()
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = json.load(f)
            EVENT_THRESHOLD = cfg.get('EVENT_THRESHOLD', EVENT_THRESHOLD)
            DRY_RUN = cfg.get('DRY_RUN', DRY_RUN)
            MAX_FARM_MASS = cfg.get('MAX_FARM_MASS', MAX_FARM_MASS)
            MAX_DRAWDOWN = cfg.get('MAX_DRAWDOWN', MAX_DRAWDOWN)
            MAX_CONCURRENT_TRADES = cfg.get('MAX_CONCURRENT_TRADES', MAX_CONCURRENT_TRADES)
            TERMINAL_PATH = cfg.get('TERMINAL_PATH', None)
            BROKER_NAME = cfg.get('BROKER_NAME', BROKER_NAME)
            print(f"[*] Loaded Config for: {BROKER_NAME} | Drawdown Limit: {MAX_DRAWDOWN}% | Max Trades: {MAX_CONCURRENT_TRADES}") 

def liquidate_farm(symbol, reason):
    """Liquidates position when tension resolves or physics invert."""
    if symbol in active_farms:
        farm = active_farms[symbol]
        mass = farm['accumulated_mass']
        ticket = farm.get('ticket')
        
        if not DRY_RUN and ticket:
            pos = mt5.positions_get(ticket=ticket)
            if pos and len(pos) > 0:
                pos = pos[0]
                tick = mt5.symbol_info_tick(symbol)
                symbol_info = mt5.symbol_info(symbol)
                order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
                price = round(price, symbol_info.digits)
                
                filling_type = mt5.ORDER_FILLING_IOC
                if symbol_info.filling_mode & 1:
                    filling_type = mt5.ORDER_FILLING_FOK
                elif symbol_info.filling_mode & 2:
                    filling_type = mt5.ORDER_FILLING_IOC
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": pos.volume,
                    "type": order_type,
                    "position": ticket,
                    "price": price,
                    "deviation": 20,
                    "magic": 100100,
                    "comment": reason[:20],
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": filling_type,
                }
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"[-] Failed to liquidate {symbol} | Error: {result.comment}")
                else:
                    print(f"[+] Successfully Liquidated MT5 Ticket #{ticket}")
                    
        ts = datetime.now().strftime('%H:%M:%S')
        print("\n" + "="*80)
        print(f"[$] AION LIQUIDATION | T_Exit: {ts}")
        print("="*80)
        print(f"Target:      [{symbol}]")
        print(f"Reason:      {reason}")
        print(f"Action:      Liquidating Total Mass: {mass} units.")
        print("="*80)
        del active_farms[symbol]

def execute_singularity(symbol, direction, risk_pct, stop_dist, ripple_target, tp_price=None):
    """Executes the physical rupture and initiates the shield."""
    if symbol not in active_farms:
        active_farms[symbol] = {'direction': direction, 'accumulated_mass': 1, 'entry_time': datetime.now().strftime('%H:%M:%S')}
        action = "PRIMARY RUPTURE ENTRY"
    else:
        if active_farms[symbol]['direction'] != direction:
            liquidate_farm(symbol, "Directional Inversion (Physics Swap)")
            return
            
        if active_farms[symbol]['accumulated_mass'] < MAX_FARM_MASS:
            active_farms[symbol]['accumulated_mass'] += 1
            action = "SCALED IN"
        else:
            return

    mass = active_farms[symbol]['accumulated_mass']
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"\n[+] {action} | T=0: {ts} | Target: [{symbol}] -> {direction} | Mass: {mass}/{MAX_FARM_MASS}")
    print(f"    THE SHIELD: Dynamic Risk: {risk_pct:.2f}% | Topographical Stop: {stop_dist:.5f} distance")
    print(f"    THE RIPPLE: Executing Shockwave initiated by [{ripple_target}]")
    if not DRY_RUN and action == "PRIMARY RUPTURE ENTRY":
        account_info = mt5.account_info()
        symbol_info = mt5.symbol_info(symbol)
        
        
        if account_info and symbol_info:
            # MARGIN SAFETY CHECKS
            total_positions = mt5.positions_total()
            if total_positions >= MAX_CONCURRENT_TRADES:
                print(f"[-] RUPTURE ABORTED: Max concurrent trades ({MAX_CONCURRENT_TRADES}) reached.")
                return
                
            if account_info.margin_level < 150.0 and account_info.margin_level != 0.0:
                print(f"[-] RUPTURE ABORTED: Margin Level too low ({account_info.margin_level}%).")
                return
                
            balance = account_info.margin_free
            risk_amount = balance * (risk_pct / 100.0)
            
            tick = mt5.symbol_info_tick(symbol)
            
            min_stop = symbol_info.trade_tick_size * 10.0
            if hasattr(symbol_info, 'trade_stops_level'):
                min_stop = max(min_stop, symbol_info.trade_stops_level * symbol_info.point * 1.2)
            
            sl_dist_abs = max(stop_dist, min_stop)
            
            if sl_dist_abs > 0 and symbol_info.trade_tick_size > 0:
                if symbol_info.trade_tick_value > 0:
                    sl_points = sl_dist_abs / symbol_info.trade_tick_size
                    loss_per_lot = sl_points * symbol_info.trade_tick_value
                    raw_lot = risk_amount / loss_per_lot
                else:
                    # Fallback if tick_value is 0 (broker data issue)
                    raw_lot = symbol_info.volume_min
                    print(f"[!] Warning: {symbol} trade_tick_value is 0. Using minimum lot size.")
                
                step = symbol_info.volume_step
                lot = round(raw_lot / step) * step
                lot = max(symbol_info.volume_min, min(lot, symbol_info.volume_max))
                
                order_type = mt5.ORDER_TYPE_BUY if 'LONG' in direction else mt5.ORDER_TYPE_SELL
                price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
                price = round(price, symbol_info.digits)
                
                sl = price - sl_dist_abs if order_type == mt5.ORDER_TYPE_BUY else price + sl_dist_abs
                sl = round(sl, symbol_info.digits)
                
                filling_type = mt5.ORDER_FILLING_IOC
                if symbol_info.filling_mode & 1:
                    filling_type = mt5.ORDER_FILLING_FOK
                elif symbol_info.filling_mode & 2:
                    filling_type = mt5.ORDER_FILLING_IOC
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": float(lot),
                    "type": order_type,
                    "price": price,
                    "sl": sl,
                    "deviation": 20,
                    "magic": 100100,
                    "comment": "AION_PRIME",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": filling_type,
                }
                if tp_price is not None:
                    request["tp"] = round(tp_price, symbol_info.digits)
                
                res = mt5.order_send(request)
                if res is None:
                    error_msg = f"[-] MT5 OrderSend Failed completely (Returned None). Error code: {mt5.last_error()}"
                    print(error_msg)
                    with open("trade_log.txt", "a") as f: f.write(f"{datetime.now()} | {error_msg}\n")
                elif res.retcode != mt5.TRADE_RETCODE_DONE:
                    error_msg = f"[-] MT5 OrderSend Failed | Code: {res.retcode} | {res.comment} | Sym: {symbol} | Vol: {lot} | Prc: {price} | SL: {sl}"
                    print(error_msg)
                    with open("trade_log.txt", "a") as f: f.write(f"{datetime.now()} | {error_msg}\n")
                else:
                    success_msg = f"[+] MT5 Order Executed | Ticket: {res.order} | Vol: {lot} | Price: {res.price}"
                    print(success_msg)
                    with open("trade_log.txt", "a") as f: f.write(f"{datetime.now()} | {success_msg}\n")
                    active_farms[symbol]['ticket'] = res.order
            else:
                error_msg = f"[-] RUPTURE ABORTED: Invalid Math | SL_Dist: {sl_dist_abs} | Tick_Size: {symbol_info.trade_tick_size}"
                print(error_msg)
                with open("trade_log.txt", "a") as f: f.write(f"{datetime.now()} | {error_msg}\n")

def run_farm_daemon():
    load_config()
    print(f"[*] AION LIVE SENTINEL ({BROKER_NAME}): Initializing Pure Singularity Engine...")
    
    init_kwargs = {}
    if TERMINAL_PATH:
        init_kwargs['path'] = TERMINAL_PATH
        
    if not mt5.initialize(**init_kwargs):
        print(f"[-] MT5 Init Failed for {BROKER_NAME}. Path: {TERMINAL_PATH}")
        return

    candidate_symbols = []
    
    VALID_ROOTS = ['EUR', 'USD', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF', 'SGD', 'ZAR', 'NOK', 'SEK', 'DKK', 'MXN', 'PLN', 'HUF', 'CZK', 'HKD', 'XAU', 'XAG', 'XPT', 'XPD']
    valid_pairs = tuple(r1 + r2 for r1 in VALID_ROOTS for r2 in VALID_ROOTS if r1 != r2)
    
    all_symbols = mt5.symbols_get()
    if all_symbols is None: return
    for sym in all_symbols:
        if sym.visible and sym.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
            # UNIVERSAL LINGUISTIC DISCOVERY ENGINE
            if sym.name.upper().startswith(valid_pairs):
                candidate_symbols.append(sym.name)
    symbols = []
    
    for sym in candidate_symbols:
        tick = mt5.symbol_info_tick(sym)
        if tick:
            symbols.append(sym)
            
    print(f"[+] Living Manifold established at {len(symbols)} Dimensions.")
    
    t_tensor = threading.Thread(target=watch_tensor, daemon=True)
    t_tensor.start()
    
    dynamic_window = 50 # Rolling window length for live KDE evaluation
    
    # Pre-allocate 2D Numpy Matrix
    prices_matrix = np.full((dynamic_window, len(symbols)), np.nan)
    
    # PRE-FILL KINETIC CACHE FROM M1 HISTORY
    for idx, sym in enumerate(symbols):
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M1, 0, dynamic_window)
        if rates is not None and len(rates) == dynamic_window:
            prices_matrix[:, idx] = rates['close']
            
    # Initial forward/backward fill for the cold boot sequence
    df_temp = pd.DataFrame(prices_matrix)
    df_temp.ffill(inplace=True)
    df_temp.bfill(inplace=True)
    prices_matrix = df_temp.values
    
    current_prices = {sym: collections.deque(maxlen=100) for sym in symbols}
    cumulative_ticks = {sym: 0 for sym in symbols}
    last_tick_time = {sym: 0 for sym in symbols}

    print(f"[*] Sentinel Online. Awaiting independent dynamic volume injection...")

    try:
        while True:
            if not DRY_RUN:
                acc = mt5.account_info()
                if acc:
                    dd_pct = ((acc.balance - acc.equity) / acc.balance) * 100.0 if acc.balance > 0 else 0
                    if dd_pct > MAX_DRAWDOWN:
                        print(f"\n[!!!] CRITICAL: EQUITY DRAWDOWN OF {dd_pct:.2f}% EXCEEDS LIMIT OF {MAX_DRAWDOWN}% [!!!]")
                        print("[!!!] INITIATING EMERGENCY LIQUIDATION OF ALL ASSETS [!!!]")
                        for sym in list(active_farms.keys()):
                            liquidate_farm(sym, "EMERGENCY MAX DRAWDOWN CROSSED")
                        
                        # Liquidate any other lingering MT5 positions not in active_farms
                        positions = mt5.positions_get()
                        if positions:
                            for pos in positions:
                                print(f"[-] Orphaned Position Found: {pos.symbol}. Liquidating...")
                                tick = mt5.symbol_info_tick(pos.symbol)
                                s_info = mt5.symbol_info(pos.symbol)
                                o_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                                p = tick.bid if o_type == mt5.ORDER_TYPE_SELL else tick.ask
                                req = {
                                    "action": mt5.TRADE_ACTION_DEAL,
                                    "symbol": pos.symbol,
                                    "volume": pos.volume,
                                    "type": o_type,
                                    "position": pos.ticket,
                                    "price": round(p, s_info.digits),
                                    "deviation": 20,
                                    "magic": 100100,
                                    "comment": "EMERGENCY DRAWDOWN CROSSED",
                                    "type_time": mt5.ORDER_TIME_GTC,
                                    "type_filling": mt5.ORDER_FILLING_IOC,
                                }
                                mt5.order_send(req)
                        time.sleep(10) # Pause daemon for 10s after lockdown
                        continue
                        
            triggered_sym = None
            
            for sym in symbols:
                tick = mt5.symbol_info_tick(sym)
                if tick and tick.time_msc > last_tick_time[sym]:
                    p = (tick.bid + tick.ask) / 2.0
                    current_prices[sym].append(p)
                    cumulative_ticks[sym] += 1
                    
                    last = last_tick_time[sym]
                    last_tick_time[sym] = tick.time_msc
                    
                    # WEEKEND CIRCUIT BREAKER
                    if last > 0 and (tick.time_msc - last) > 12 * 3600 * 1000:
                        print(f"[!] WEEKEND CIRCUIT BREAKER TRIPPED by {sym}! Delta: {(tick.time_msc - last)/(3600*1000):.2f} hours")
                        print("[!] Flushing Matrix and Refetching Historical Anchor states...")
                        
                        # Rebuild Memory Matrix
                        for i_idx, i_sym in enumerate(symbols):
                            rates = mt5.copy_rates_from_pos(i_sym, mt5.TIMEFRAME_M1, 0, dynamic_window)
                            if rates is not None and len(rates) == dynamic_window:
                                prices_matrix[:, i_idx] = rates['close']
                        df_temp = pd.DataFrame(prices_matrix)
                        df_temp.ffill(inplace=True)
                        df_temp.bfill(inplace=True)
                        prices_matrix = df_temp.values
                        
                        cumulative_ticks = {s: 0 for s in symbols}
                        current_prices = {s: collections.deque(maxlen=100) for s in symbols}
                        last_tick_time = {s: tick.time_msc for s in symbols}
                    
                    threshold = OPTIMAL_THRESHOLDS.get(sym, 2500)
                    if cumulative_ticks[sym] >= threshold:
                        triggered_sym = sym
            
            if triggered_sym:
                cumulative_ticks[triggered_sym] = 0
                
                new_state = np.zeros(len(symbols))
                for idx, sym in enumerate(symbols):
                    if current_prices[sym]:
                        new_state[idx] = np.mean(current_prices[sym])
                    else:
                        new_state[idx] = prices_matrix[-1, idx] # Instant O(1) Numpy forward-fill
                
                # Shift matrix up by 1 and insert new state at the bottom (Ring Buffer)
                prices_matrix = np.roll(prices_matrix, -1, axis=0)
                prices_matrix[-1, :] = new_state
                
                print(f"[~] Event State Captured via {triggered_sym}. Mapping Topology...")
                
                if True:
                    prices = prices_matrix.astype(float)
                    
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
                    
                    alphas = np.zeros(len(symbols))
                    valid_traps = []
                    kdes = {}
                    
                    for idx in range(len(symbols)):
                        hist_p = prices[:, idx]
                        p_min, p_max = np.min(hist_p), np.max(hist_p)
                        if p_max == p_min: continue
                            
                        try:
                            kde = gaussian_kde(hist_p, bw_method='silverman')
                            kdes[idx] = kde
                        except np.linalg.LinAlgError:
                            continue
                            
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
                        alpha_val = (norm_force * (local_density / avg_density)) / ((void_density / avg_density) + 1e-6)
                        alphas[idx] = alpha_val
                        
                        if np.sign(network_force[idx]) != np.sign(current_v[idx]) or abs(current_v[idx]) < 0.1:
                            valid_traps.append(idx)
                    
                    mean_alpha = np.mean(alphas)
                    std_alpha = np.std(alphas) + 1e-6
                    
                    for trap_idx in valid_traps:
                        node_alpha = alphas[trap_idx]
                        
                        # 3 Sigma Event Threshold
                        if node_alpha > mean_alpha + (3.0 * std_alpha) and node_alpha > 5.0:
                            trap_sym = symbols[trap_idx]
                            push_up = network_force[trap_idx] > 0
                            direction = "LONG (BUY)" if push_up else "SHORT (SELL)"
                            
                            # THE FRICTION FILTER
                            hist_p = prices[:, trap_idx]
                            tick = mt5.symbol_info_tick(trap_sym)
                            if tick:
                                spread = tick.ask - tick.bid
                                friction = spread * 1.5
                                vacuum_dist = abs(prices[-1, trap_idx] - np.mean(hist_p))
                                if vacuum_dist <= friction:
                                    print(f"[-] RUPTURE ABORTED [{trap_sym}]: Potential Energy ({vacuum_dist:.5f}) <= Friction ({friction:.5f})")
                                    continue
                            
                            # THE SHIELD: Dynamic Risk & Topographical Stop
                            kde = kdes[trap_idx]
                            hist_p = prices[:, trap_idx]
                            eval_p = np.linspace(np.min(hist_p), np.max(hist_p), 100)
                            density = kde(eval_p)
                            
                            peaks, _ = find_peaks(density)
                            if len(peaks) > 0:
                                main_wall_idx = peaks[np.argmax(density[peaks])]
                                wall_price = eval_p[main_wall_idx]
                            else:
                                wall_price = prices[-1, trap_idx]
                            
                            p_range = np.max(hist_p) - np.min(hist_p)
                            if push_up:
                                stop_dist = (prices[-1, trap_idx] - (wall_price - p_range * 0.2))
                            else:
                                stop_dist = ((wall_price + p_range * 0.2) - prices[-1, trap_idx])
                            stop_dist = max(stop_dist, 0.001)
                            
                            sigma_dev = (node_alpha - mean_alpha) / std_alpha
                            risk_pct = min(5.0, max(1.0, 1.0 + (sigma_dev - 3.0) * 2.0))
                            
                            # THE RIPPLE: Secondary Correlation
                            corr_row = adj[trap_idx].copy()
                            corr_row[trap_idx] = -1 
                            secondary_idx = np.argmax(corr_row)
                            ripple_sym = symbols[secondary_idx]
                            
                            ripple_tick = mt5.symbol_info_tick(ripple_sym)
                            vacuum_dist = 0.0
                            friction = 0.0
                            if ripple_tick:
                                ripple_spread = ripple_tick.ask - ripple_tick.bid
                                friction = ripple_spread * 1.5
                                
                                ripple_hist_p = prices[:, secondary_idx]
                                ripple_curr_p = prices[-1, secondary_idx]
                                ripple_barycenter = np.mean(ripple_hist_p)
                                vacuum_dist = abs(ripple_curr_p - ripple_barycenter)
                                
                                if vacuum_dist <= friction:
                                    print(f"[-] RUPTURE ABORTED [{ripple_sym}]: Ripple Potential Energy ({vacuum_dist:.5f}) <= Friction ({friction:.5f})")
                                    continue
                                    
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
                                    stop_dist_abs = abs(ripple_curr_p - r_stop)
                                else:
                                    r_stop = ripple_p_max + (ripple_p_range * 0.2)
                                    if r_stop - ripple_curr_p < friction * 2:
                                        r_stop = ripple_curr_p + friction * 2
                                    stop_dist_abs = abs(r_stop - ripple_curr_p)
                                
                                execute_singularity(ripple_sym, r_dir, risk_pct, stop_dist_abs, trap_sym, ripple_barycenter)
                
                pass
            
            time.sleep(0.01) 
            
    except KeyboardInterrupt:
        print("\n[!] Live Sentinel Shutdown Requested. Retracting Shields...")
        for sym in list(active_farms.keys()):
            liquidate_farm(sym, "Manual Daemon Shutdown")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    run_farm_daemon()
