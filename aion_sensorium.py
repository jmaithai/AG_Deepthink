import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.simplefilter(action='ignore')

def build_aion_sensorium():
    print("[*] AION SENSORIUM: Initializing Barless Market Observatory...")
    if not mt5.initialize():
        print("[-] MT5 Init Failed.")
        return

    # A contained universe as per the AION Doctrine
    target_symbols = [
        "XAUUSD", "XAGUSD", "XAUEUR", "XAUGBP", "XAUJPY", 
        "EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCAD", "CADJPY"
    ]

    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24) # 24-hour raw tick capture
    
    print(f"[*] Fetching continuous raw tick stream...")
    
    raw_streams = []
    active_symbols = []
    
    for sym in target_symbols:
        info = mt5.symbol_info(sym)
        if info is None: continue
        
        mt5.symbol_select(sym, True)
        ticks = mt5.copy_ticks_range(sym, start_time, end_time, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            print(f" [!] Missing ticks for {sym}. Skipping.")
            continue
            
        df = pd.DataFrame(ticks)
        df['time'] = pd.to_datetime(df['time_msc'], unit='ms')
        df.set_index('time', inplace=True)
        
        temp_df = pd.DataFrame(index=df.index)
        temp_df[f"{sym}"] = (df['bid'] + df['ask']) / 2.0
        
        # Aggregate duplicate millisecond timestamps
        temp_df = temp_df.groupby(temp_df.index).last()
        raw_streams.append(temp_df)
        active_symbols.append(sym)
        print(f" [+] {sym}: {len(df):,} discrete events captured.")

    mt5.shutdown()

    if not raw_streams:
        print("[-] Fatal: No tick stream retrieved.")
        return

    print("\n[*] Eradicating Chronological Time...")
    master_df = pd.concat(raw_streams, axis=1, join='outer')
    master_df.ffill(inplace=True)
    master_df.dropna(inplace=True)
    
    # ---------------------------------------------------------
    # 1. THE EVENT CLOCK
    # ---------------------------------------------------------
    print("[*] Constructing Event Packets (Constant Network Activity Buckets)...")
    master_df['cumulative_ticks'] = np.arange(len(master_df))
    
    # The clock ticks ONLY when 5,000 physical network transactions occur
    EVENT_THRESHOLD = 5000 
    master_df['Event_Clock'] = master_df['cumulative_ticks'] // EVENT_THRESHOLD
    master_df['chronological_time'] = master_df.index
    
    sensorium = master_df.groupby('Event_Clock').agg(
        Chronological_Start=('chronological_time', 'first'),
        Chronological_End=('chronological_time', 'last'),
        Ticks_Consumed=('cumulative_ticks', 'count')
    )
    
    for col in active_symbols:
        sensorium[col] = master_df.groupby('Event_Clock')[col].last()
        
    print(f"[+] Compressed {len(master_df):,} chronological events into {len(sensorium):,} True Phase States.")
    
    # ---------------------------------------------------------
    # 2. TEMPERATURE & KINEMATICS (The Physics)
    # ---------------------------------------------------------
    sensorium['dt_seconds'] = (sensorium['Chronological_End'] - sensorium['Chronological_Start']).dt.total_seconds()
    sensorium['dt_seconds'] = sensorium['dt_seconds'].replace(0, 0.001) 
    
    sensorium['Temperature'] = sensorium['Ticks_Consumed'] / sensorium['dt_seconds']
    avg_temp = sensorium['Temperature'].mean()
    
    prices = np.log(sensorium[active_symbols].values + 1e-9)
    velocity = np.diff(prices, axis=0, prepend=prices[0:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
    jerk = np.diff(acceleration, axis=0, prepend=acceleration[0:1]) # The transition snap
    
    sensorium['System_Jerk'] = np.linalg.norm(jerk, axis=1)
    
    # ---------------------------------------------------------
    # 3. REALITY DEGENERACY (Wave-Function Collapse)
    # ---------------------------------------------------------
    print("[*] Inferring Hidden States (Eigen-Decomposition)...")
    window = 10 
    degeneracy_records = []
    
    for i in range(window, len(sensorium)):
        v_window = velocity[i-window:i]
        
        # Z-Score normalization to find pure geometric correlation
        mean_v = np.mean(v_window, axis=0)
        std_v = np.std(v_window, axis=0) + 1e-12
        v_norm = (v_window - mean_v) / std_v
        
        v_noisy = v_norm + np.random.normal(0, 1e-12, v_norm.shape)
        cov_matrix = np.cov(v_noisy.T)
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        total_energy = np.sum(eigenvalues) + 1e-12
        p_energy = np.clip(eigenvalues / total_energy, 1e-12, 1.0)
        
        # Shannon Entropy = Reality Degeneracy
        S_degeneracy = -np.sum(p_energy * np.log2(p_energy))
        
        degeneracy_records.append({
            'Event_State': i,
            'Reality_Degeneracy': S_degeneracy,
            'Dominant_Power': p_energy[0],
            'Dominant_Vector': eigenvectors[:, 0]
        })
        
    results = pd.DataFrame(degeneracy_records)
    results['Degeneracy_Drop'] = results['Reality_Degeneracy'].diff().fillna(0)
    
    # Find the single most violent wave-function collapse
    eruption_idx = results['Degeneracy_Drop'].idxmin()
    eruption_event = results.loc[eruption_idx]
    
    state_idx = int(eruption_event['Event_State'])
    
    e_start = sensorium['Chronological_Start'].iloc[state_idx].strftime('%Y-%m-%d %H:%M:%S')
    e_end = sensorium['Chronological_End'].iloc[state_idx].strftime('%Y-%m-%d %H:%M:%S')
    e_temp = sensorium['Temperature'].iloc[state_idx]
    e_jrk = sensorium['System_Jerk'].iloc[state_idx]
    
    vec = eruption_event['Dominant_Vector']
    loadings = {active_symbols[j]: vec[j] for j in range(len(active_symbols))}
    sorted_loadings = sorted(loadings.items(), key=lambda item: abs(item[1]), reverse=True)
    
    origin_sym = sorted_loadings[0][0]
    receiver_sym = sorted_loadings[1][0]
    lagging_sym = sorted_loadings[-1][0]
    lagging_sym2 = sorted_loadings[-2][0]
    
    direction = "Compression transitioning into release" if e_jrk > sensorium['System_Jerk'].mean() else "Absorption"
    temp_state = "Heating (High Agitation)" if e_temp > avg_temp else "Cooling (Compression)"
    
    print("\n" + "="*80)
    print("AION ENGINE: LIVING THESIS")
    print("="*80)
    
    print(f"MARKET STATE:")
    print(f"{direction}. High structural jerk ({e_jrk:.4f}) detected.")
    print(f"Temperature: {e_temp:.2f} ticks/sec | {temp_state}")
    print(f"Chronological boundary: {e_start} -> {e_end}\n")
    
    print(f"DOMINANT WHY:")
    print(f"Pressure wave originating in [{origin_sym}], breaking local equilibria.")
    
    print(f"\nREALITY DEGENERACY:")
    print(f"Falling. Degeneracy dropped by {abs(eruption_event['Degeneracy_Drop']):.4f} bits.")
    print(f"Explanations collapsing. A single hidden state now commands {eruption_event['Dominant_Power']*100:.1f}% of network variance.")
    
    print(f"\nPROPAGATION:")
    print(f"Epicenter: [{origin_sym}] (Eigen-Weight: {sorted_loadings[0][1]:+.4f})")
    print(f"Wavefront Receiver: [{receiver_sym}] (Eigen-Weight: {sorted_loadings[1][1]:+.4f})")
    print(f"Dead / Lagging: [{lagging_sym}] and [{lagging_sym2}]")
    
    print(f"\nTRAP & WEAKEST EXIT:")
    print(f"Participants applying chronological timeframes to [{origin_sym}] are trapped in phase transition.")
    print(f"Weakest Outlet: Asynchronous receivers lagging the wave: [{lagging_sym}].")
    
    print(f"\nACTION PHYSICS ARBITER:")
    if eruption_event['Reality_Degeneracy'] > 2.5:
        print(f"Condition: Too many explanations alive. (Entropy = {eruption_event['Reality_Degeneracy']:.2f})")
        print(f"Action: NULL_AVOID.")
    else:
        print(f"Condition: Asymmetric route forming.")
        print(f"Action: Monitor [{lagging_sym}] for confirmation of forced liquidity sweep.")
        print(f"NULL_AVOID: If [{lagging_sym}] fails to absorb the shockwave, abort.")
    print("="*80)

if __name__ == "__main__":
    build_aion_sensorium()
