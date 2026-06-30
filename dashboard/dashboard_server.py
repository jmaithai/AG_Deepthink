import asyncio
import websockets
import json
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import threading
import http.server
import socketserver
import argparse
import os
import collections
from datetime import datetime
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

import warnings
warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

HTTP_PORT = 8080
WS_PORT = 9080
TERMINAL_PATH = None
BROKER_NAME = "DEFAULT"
DRY_RUN = True
EVENT_THRESHOLD = 25000

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


active_symbols = []

# Shadow Ledger State
shadow_trades = {}
trade_id_counter = 0
closed_shadow_trades = []
shadow_balance = 200.0
LEDGER_FILE = "shadow_ledger.json"

def load_ledger():
    global shadow_balance, closed_shadow_trades, trade_id_counter
    if os.path.exists(LEDGER_FILE):
        try:
            with open(LEDGER_FILE, 'r') as f:
                data = json.load(f)
                shadow_balance = data.get('shadow_balance', 200.0)
                closed_shadow_trades = data.get('closed_shadow_trades', [])
                if closed_shadow_trades:
                    trade_id_counter = max([t.get('id', 0) for t in closed_shadow_trades])
        except Exception as e:
            print(f"[!] Error loading ledger: {e}")

def save_ledger():
    global shadow_balance, closed_shadow_trades
    try:
        with open(LEDGER_FILE, 'w') as f:
            json.dump({
                'shadow_balance': shadow_balance,
                'closed_shadow_trades': closed_shadow_trades
            }, f, indent=4)
    except Exception as e:
        print(f"[!] Error saving ledger: {e}")

def serve_http():
    dashboard_dir = os.path.dirname(os.path.abspath(__file__))
    
    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=dashboard_dir, **kwargs)
            
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", HTTP_PORT), CustomHandler) as httpd:
            print(f"[*] Dashboard UI available at: http://localhost:{HTTP_PORT}")
            httpd.serve_forever()
    except Exception as e:
        print(f"[!] HTTP Server failed on {HTTP_PORT}: {e}")

async def physics_stream(websocket):
    global active_symbols, shadow_trades, closed_shadow_trades, shadow_balance, trade_id_counter
    print("[*] Dashboard UI Connected via WebSocket.")
    
    dynamic_window = 50
    last_msc = {sym: 0 for sym in active_symbols}
    
    # Pre-allocate 2D Numpy Matrix
    prices_matrix = np.full((dynamic_window, len(active_symbols)), np.nan)
    
    # PRE-FILL KINETIC CACHE FROM M1 HISTORY
    for idx, sym in enumerate(active_symbols):
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M1, 0, dynamic_window)
        if rates is not None and len(rates) == dynamic_window:
            prices_matrix[:, idx] = rates['close']
            
    # Initial forward/backward fill for the cold boot sequence
    df_temp = pd.DataFrame(prices_matrix)
    df_temp.ffill(inplace=True)
    df_temp.bfill(inplace=True)
    prices_matrix = df_temp.values

    # Use rolling window of 100 ticks to calculate localized smoothed price
    current_prices = {sym: collections.deque(maxlen=100) for sym in active_symbols}
    cumulative_ticks = {sym: 0 for sym in active_symbols}
    
    while True:
        try:
            if not active_symbols:
                await asyncio.sleep(1)
                continue
                
            triggered_sym = None
            
            for sym in active_symbols:
                tick = mt5.symbol_info_tick(sym)
                if tick and tick.time_msc > last_msc[sym]:
                    p = (tick.bid + tick.ask) / 2.0
                    current_prices[sym].append(p)
                    cumulative_ticks[sym] += 1
                    
                    last = last_msc[sym]
                    last_msc[sym] = tick.time_msc
                    
                    # WEEKEND CIRCUIT BREAKER
                    if last > 0 and (tick.time_msc - last) > 12 * 3600 * 1000:
                        print(f"[!] WEEKEND CIRCUIT BREAKER TRIPPED by {sym}! Delta: {(tick.time_msc - last)/(3600*1000):.2f} hours")
                        print("[!] Flushing Matrix and Refetching Historical Anchor states...")
                        
                        # Rebuild Memory Matrix
                        for i_idx, i_sym in enumerate(active_symbols):
                            rates = mt5.copy_rates_from_pos(i_sym, mt5.TIMEFRAME_M1, 0, dynamic_window)
                            if rates is not None and len(rates) == dynamic_window:
                                prices_matrix[:, i_idx] = rates['close']
                        df_temp = pd.DataFrame(prices_matrix)
                        df_temp.ffill(inplace=True)
                        df_temp.bfill(inplace=True)
                        prices_matrix = df_temp.values
                        
                        cumulative_ticks = {s: 0 for s in active_symbols}
                        current_prices = {s: collections.deque(maxlen=100) for s in active_symbols}
                        last_msc = {s: tick.time_msc for s in active_symbols}
                    
                    threshold = OPTIMAL_THRESHOLDS.get(sym, EVENT_THRESHOLD)
                    if cumulative_ticks[sym] >= threshold:
                        triggered_sym = sym
            
            # Send heartbeat progress of the fastest ticking node relative to its threshold
            max_progress = 0
            for sym in active_symbols:
                prog = cumulative_ticks[sym] / OPTIMAL_THRESHOLDS.get(sym, EVENT_THRESHOLD)
                if prog > max_progress:
                    max_progress = prog
                    
            heartbeat = {
                'type': 'heartbeat',
                'volume_progress': min(max_progress * 100, 100)
            }
            await websocket.send(json.dumps(heartbeat))
            
            # If any independent dimension hits its threshold, evaluate the global matrix
            if triggered_sym:
                # Reset only the triggering node's clock
                cumulative_ticks[triggered_sym] = 0
                
                # Snapshot global structural state using smoothed localized prices
                new_state = np.zeros(len(active_symbols))
                for idx, sym in enumerate(active_symbols):
                    if current_prices[sym]:
                        new_state[idx] = np.mean(current_prices[sym])
                    else:
                        new_state[idx] = prices_matrix[-1, idx] # Instant O(1) Numpy forward-fill
                
                # Shift matrix up by 1 and insert new state at the bottom (Ring Buffer)
                prices_matrix = np.roll(prices_matrix, -1, axis=0)
                prices_matrix[-1, :] = new_state
                
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
                    
                    # SHADOW TRADING EVALUATION (Exits)
                    for t_id in list(shadow_trades.keys()):
                        trade = shadow_trades[t_id]
                        sym = trade['symbol']
                        if sym in active_symbols:
                            idx = active_symbols.index(sym)
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
                                trade['exit_price'] = curr_p
                                trade['pnl'] = pnl
                                trade['exit_time'] = datetime.now().strftime('%H:%M:%S')
                                shadow_balance += pnl
                                closed_shadow_trades.append(trade)
                                del shadow_trades[t_id]
                                save_ledger()

                    alphas = np.zeros(len(active_symbols))
                    nodes_payload = []
                    events_payload = []
                    edges_payload = []
                    
                    # Compute edges
                    for i in range(len(active_symbols)):
                        for j in range(i+1, len(active_symbols)):
                            weight = adj[i, j]
                            if weight > 0.5:
                                edges_payload.append({
                                    'source': active_symbols[i],
                                    'target': active_symbols[j],
                                    'weight': float(weight)
                                })
                    
                    for idx in range(len(active_symbols)):
                        sym = active_symbols[idx]
                        hist_p = prices[:, idx]
                        p_min, p_max = np.min(hist_p), np.max(hist_p)
                        
                        if p_max == p_min: 
                            continue
                            
                        try:
                            kde = gaussian_kde(hist_p, bw_method='silverman')
                        except np.linalg.LinAlgError:
                            continue
                            
                        curr_p = prices[-1, idx]
                        local_density = kde(curr_p)[0]
                        
                        eval_points = np.linspace(p_min, p_max, 50)
                        density_array = kde(eval_points)
                        avg_density = np.mean(density_array) + 1e-6
                        
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
                        
                    mean_alpha = np.mean(alphas)
                    std_alpha = np.std(alphas) + 1e-6
                    
                    for idx in range(len(active_symbols)):
                        sym = active_symbols[idx]
                        sigma_dev = (alphas[idx] - mean_alpha) / std_alpha
                        
                        hist_p = prices[:, idx]
                        p_min, p_max = np.min(hist_p), np.max(hist_p)
                        curr_p = prices[-1, idx]
                        
                        kde_x = []
                        kde_y = []
                        if p_max > p_min:
                            try:
                                kde = gaussian_kde(hist_p, bw_method='silverman')
                                e_pts = np.linspace(p_min - (p_max-p_min)*0.1, p_max + (p_max-p_min)*0.1, 30)
                                d_array = kde(e_pts)
                                kde_x = e_pts.tolist()
                                kde_y = d_array.tolist()
                            except:
                                pass
                        
                        nodes_payload.append({
                            'symbol': sym,
                            'sigma': float(sigma_dev),
                            'force': float(network_force[idx]),
                            'price': float(curr_p),
                            'kde_x': kde_x,
                            'kde_y': kde_y
                        })
                        
                        if sigma_dev > 3.0 and alphas[idx] > 5.0:
                            if np.sign(network_force[idx]) != np.sign(current_v[idx]) or abs(current_v[idx]) < 0.1:
                                push_up = network_force[idx] > 0
                                direction = "LONG" if push_up else "SHORT"
                                
                                # PRIMARY FRICTION FILTER
                                tick = mt5.symbol_info_tick(sym)
                                if tick:
                                    spread = tick.ask - tick.bid
                                    friction = spread * 1.5
                                    vacuum_dist = abs(curr_p - np.mean(hist_p))
                                    if vacuum_dist <= friction:
                                        continue
                                
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
                                        continue
                                        
                                    # Prevent duplicate exposure on the same symbol
                                    already_active = False
                                    for t_val in shadow_trades.values():
                                        if t_val['symbol'] == ripple_sym:
                                            already_active = True
                                            break
                                    if already_active:
                                        continue
                                        
                                    # OPEN SHADOW TRADE
                                    risk_amt = shadow_balance * (risk_pct / 100.0)
                                    trade_id_counter += 1
                                    
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
                                        'resolution': OPTIMAL_THRESHOLDS.get(ripple_sym, EVENT_THRESHOLD), # Tag with specific threshold
                                        'symbol': ripple_sym,
                                        'direction': r_dir,
                                        'entry_price': ripple_curr_p,
                                        'stop_price': r_stop,
                                        'barycenter': ripple_barycenter,
                                        'risk_amount': risk_amt,
                                        'entry_time': datetime.now().strftime('%H:%M:%S')
                                    }
                                
                                events_payload.append({
                                    'symbol': sym,
                                    'resolution': OPTIMAL_THRESHOLDS.get(sym, EVENT_THRESHOLD),
                                    'direction': direction,
                                    'sigma': float(sigma_dev),
                                    'risk': float(risk_pct),
                                    'stop_dist': float(stop_dist),
                                    'ripple_target': ripple_sym,
                                    'vacuum': float(vacuum_dist),
                                    'friction': float(friction),
                                    'timestamp': datetime.now().strftime('%H:%M:%S')
                                })
                                
                    acc_balance = shadow_balance
                    active_t = list(shadow_trades.values())
                    
                    if not DRY_RUN:
                        acc = mt5.account_info()
                        if acc:
                            acc_balance = acc.balance
                        
                        live_t = []
                        positions = mt5.positions_get()
                        if positions:
                            for pos in positions:
                                live_t.append({
                                    'symbol': pos.symbol,
                                    'direction': 'LONG' if pos.type == mt5.ORDER_TYPE_BUY else 'SHORT',
                                    'entry_price': pos.price_open,
                                    'barycenter': pos.price_open, 
                                    'risk_amount': pos.profit 
                                })
                        active_t = live_t

                    payload = {
                        'type': 'physics_update',
                        'is_live': not DRY_RUN,
                        'nodes': nodes_payload,
                        'events': events_payload,
                        'edges': edges_payload,
                        'shadow_balance': float(acc_balance),
                        'active_shadows': active_t,
                        'closed_shadows': closed_shadow_trades[-10:]
                    }
                    await websocket.send(json.dumps(payload))
                
                pass
            
            await asyncio.sleep(0.01)
            
        except websockets.exceptions.ConnectionClosed:
            print("[-] Dashboard Disconnected.")
            break
        except Exception as e:
            print(f"[!] Error: {e}")
            await asyncio.sleep(1)

async def main():
    global TERMINAL_PATH, BROKER_NAME, active_symbols, HTTP_PORT, WS_PORT, DRY_RUN
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args, _ = parser.parse_known_args()
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = json.load(f)
            TERMINAL_PATH = cfg.get('TERMINAL_PATH', None)
            BROKER_NAME = cfg.get('BROKER_NAME', BROKER_NAME)
            DRY_RUN = cfg.get('DRY_RUN', DRY_RUN)
            EVENT_THRESHOLD = cfg.get('EVENT_THRESHOLD', EVENT_THRESHOLD)
            if 'DASHBOARD_PORT' in cfg:
                HTTP_PORT = cfg['DASHBOARD_PORT']
                WS_PORT = HTTP_PORT + 1000
            
    load_ledger()
            
    print(f"[*] Starting 38-D Optimized Telemetry Bridge for {BROKER_NAME}...")
    
    init_kwargs = {}
    if TERMINAL_PATH:
        init_kwargs['path'] = TERMINAL_PATH
        
    if not mt5.initialize(**init_kwargs):
        print(f"[-] MT5 Init Failed for {BROKER_NAME}. Path: {TERMINAL_PATH}")
        return
        
    active_symbols = []
    
    VALID_ROOTS = ['EUR', 'USD', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF', 'SGD', 'ZAR', 'NOK', 'SEK', 'DKK', 'MXN', 'PLN', 'HUF', 'CZK', 'HKD', 'XAU', 'XAG', 'XPT', 'XPD']
    valid_pairs = tuple(r1 + r2 for r1 in VALID_ROOTS for r2 in VALID_ROOTS if r1 != r2)
    
    candidate_symbols = []
    all_symbols = mt5.symbols_get()
    if all_symbols:
        for sym in all_symbols:
            if sym.visible and sym.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                if sym.name.upper().startswith(valid_pairs):
                    candidate_symbols.append(sym.name)
                    
    for sym in candidate_symbols:
        tick = mt5.symbol_info_tick(sym)
        if tick:
            active_symbols.append(sym)
                    
    print(f"[+] Hooked {len(active_symbols)} dimensions with Independent Physics Clocks.")
        
    t_tensor = threading.Thread(target=watch_tensor, daemon=True)
    t_tensor.start()
        
    t = threading.Thread(target=serve_http, daemon=True)
    t.start()
    
    async with websockets.serve(physics_stream, "localhost", WS_PORT):
        print(f"[*] AION Premium Telemetry WS active on ws://localhost:{WS_PORT}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
