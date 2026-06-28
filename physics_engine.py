import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# =====================================================================
# MODULE 1: THE THERMODYNAMIC GAUGE INVERTER
# =====================================================================
CORE_NODES = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF', 'XAU', 'XAG']
ENERGY_THRESHOLD = 25000  # Network tick volume required to trigger 1 "Proper Time" tick
MAX_M1_BARS = 15000       # Raw chronological M1 bars to pull for thermodynamic compression

def initialize_mt5():
    print("[*] Initializing MetaTrader 5...")
    if not mt5.initialize():
        print(f"[-] MT5 Initialization failed. Error code: {mt5.last_error()}")
        quit()
    print("[+] MT5 Sensor Array Online.")

def discover_topology(data_source=None):
    if data_source is not None:
        return _derive_topology_from_df(data_source)
        
    print("[*] Scanning MT5 Broker Topology...")
    symbols = mt5.symbols_get()
    if symbols is None:
        print("[-] Failed to retrieve symbols.")
        return [], []

    active_edges = []
    active_nodes = set()

    for sym in symbols:
        name = sym.name.upper()
        matches = [node for node in CORE_NODES if node in name]
        
        if len(matches) == 2:
            idx0 = name.find(matches[0])
            idx1 = name.find(matches[1])
            
            if idx0 < idx1:
                base, quote = matches[0], matches[1]
            else:
                base, quote = matches[1], matches[0]
            
            mt5.symbol_select(sym.name, True)
            
            if not any(e['base'] == base and e['quote'] == quote for e in active_edges):
                active_edges.append({'symbol': sym.name, 'base': base, 'quote': quote})
                active_nodes.update([base, quote])

    active_nodes = sorted(list(active_nodes))
    print(f"[+] Discovered {len(active_nodes)} Nodes and {len(active_edges)} Edges.")
    return active_nodes, active_edges

def _derive_topology_from_df(df):
    active_edges = []
    active_nodes = set()
    
    symbols = []
    for col in df.columns:
        if col.endswith('_price'):
            symbols.append(col.replace('_price', ''))
            
    if not symbols:
        # Fallback for raw data_loader.py single-asset parquet
        symbols = ['XAUUSD']
        
    for name in symbols:
        name = name.upper()
        matches = [node for node in CORE_NODES if node in name]
        if len(matches) == 2:
            idx0 = name.find(matches[0])
            idx1 = name.find(matches[1])
            if idx0 < idx1:
                base, quote = matches[0], matches[1]
            else:
                base, quote = matches[1], matches[0]
            
            active_edges.append({'symbol': name, 'base': base, 'quote': quote})
            active_nodes.update([base, quote])
            
    active_nodes = sorted(list(active_nodes))
    print(f"[+] Discovered {len(active_nodes)} Nodes and {len(active_edges)} Edges from historical data.")
    return active_nodes, active_edges

def fetch_and_warp_fluid(active_edges, data_source=None):
    if data_source is not None:
        print(f"[*] Processing historical fluid dynamics for {len(active_edges)} edges...")
        df = data_source.copy()
        
        # If it's a raw MT5 dataframe (single asset) from data_loader.py
        if 'close' in df.columns and len(active_edges) == 1:
            sym = active_edges[0]['symbol']
            if 'time' in df.columns:
                df.set_index('time', inplace=True)
            temp_df = pd.DataFrame(index=df.index)
            temp_df[f"{sym}_price"] = np.log(df['close'])
            temp_df[f"{sym}_vol"] = df['tick_volume']
            master_df = temp_df
        else:
            # Assume it's already a formatted master_df (multi-asset)
            master_df = df
            
    else:
        print(f"[*] Downloading fluid dynamics for {len(active_edges)} edges...")
        df_list = []
        
        for edge in active_edges:
            sym = edge['symbol']
            rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M1, 0, MAX_M1_BARS)
            if rates is None or len(rates) == 0:
                continue
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            temp_df = pd.DataFrame(index=df.index)
            temp_df[f"{sym}_price"] = np.log(df['close'])
            temp_df[f"{sym}_vol"] = df['tick_volume']
            
            df_list.append(temp_df)
    
        print("[*] Synchronizing network tensor...")
        master_df = pd.concat(df_list, axis=1)
        
    master_df.ffill(inplace=True)
    master_df.dropna(inplace=True)

    print("[*] Warping Chronological Time into Thermodynamic States (τ)...")
    vol_cols = [col for col in master_df.columns if col.endswith('_vol')]
    master_df['network_kinetic_energy'] = master_df[vol_cols].sum(axis=1)
    master_df['cumulative_energy'] = master_df['network_kinetic_energy'].cumsum()
    master_df['Energy_State'] = (master_df['cumulative_energy'] // ENERGY_THRESHOLD).astype(int)
    
    price_cols = [col for col in master_df.columns if col.endswith('_price')]
    thermo_df = master_df.groupby('Energy_State')[price_cols].last()
    
    print(f"[+] Compressed {len(master_df)} chronological minutes into {len(thermo_df)} Proper Time states.")
    return thermo_df

def extract_hoberman_space(thermo_df, active_nodes, active_edges):
    print("[*] Inverting Gauge to extract Hoberman Pressures...")
    N = len(active_nodes)
    E = len(active_edges)
    
    node_idx = {node: i for i, node in enumerate(active_nodes)}
    B = np.zeros((E, N))
    edge_symbols = []
    
    for i, edge in enumerate(active_edges):
        sym = edge['symbol']
        edge_symbols.append(f"{sym}_price")
        
        b_idx = node_idx[edge['base']]
        q_idx = node_idx[edge['quote']]
        
        B[i, b_idx] = 1.0
        B[i, q_idx] = -1.0

    B_constrained = np.vstack([B, np.ones(N)])
    pressure_history = np.zeros((len(thermo_df), N))
    prices_array = thermo_df[edge_symbols].values
    
    for t in range(len(thermo_df)):
        log_prices = prices_array[t]
        Y_constrained = np.append(log_prices, 0.0)
        P, _, _, _ = np.linalg.lstsq(B_constrained, Y_constrained, rcond=None)
        pressure_history[t] = P

    pressure_df = pd.DataFrame(pressure_history, columns=active_nodes, index=thermo_df.index)
    return pressure_df

def main():
    initialize_mt5()
    active_nodes, active_edges = discover_topology()
    
    if not active_nodes:
        print("[-] No network topology found. Exiting.")
        mt5.shutdown()
        return
        
    thermo_df = fetch_and_warp_fluid(active_edges)
    if thermo_df.empty:
        print("[-] Not enough fluid dynamics downloaded. Exiting.")
        mt5.shutdown()
        return
        
    pressure_df = extract_hoberman_space(thermo_df, active_nodes, active_edges)
    mt5.shutdown()
    
    print("\n" + "="*60)
    print("CURRENT ABSOLUTE NODE PRESSURES (LATEST ENERGY STATE)")
    print("="*60)
    latest_P = pressure_df.iloc[-1]
    print(latest_P.to_string())
    print("-" * 60)
    
    sum_P = latest_P.sum()
    print(f"GAUGE CHECKSUM (Must be exactly 0.0): {sum_P:.15f}")
    
    if abs(sum_P) < 1e-10:
        print("[SUCCESS] The topological manifold is closed and mathematically perfect. 🟢")
    else:
        print("[FATAL] Energy leak detected. The matrix is structurally compromised. 🔴")

if __name__ == "__main__":
    main()
