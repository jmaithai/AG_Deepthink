import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_topography_mapper():
    file_path = "archive/fractal_manifold_6M.parquet"
    print(f"[*] AION TOPOGRAPHY MAPPER: Ingesting 6-Month Physical Field from {file_path}")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Critical Error: {e}")
        return

    # Gauge Inversion Fixes
    close_cols = [c for c in df.columns if c.endswith('_M1_price')]
    vol_cols = [c for c in df.columns if c.endswith('_M1_vol')]
    symbols = [c.replace('_M1_price', '') for c in close_cols]

    print("[*] Eradicating Chronological Time...")
    df['network_ticks'] = df[vol_cols].sum(axis=1)
    df['cumulative_ticks'] = df['network_ticks'].cumsum()
    EVENT_THRESHOLD = int(df['network_ticks'].mean() * 1000) 
    df['Event_Clock'] = (df['cumulative_ticks'] // EVENT_THRESHOLD).astype(int)
    
    sensorium = df.groupby('Event_Clock').last()
    
    if 'time' in df.columns:
        sensorium['chronological_time'] = df.groupby('Event_Clock')['time'].last()
    else:
        sensorium['chronological_time'] = df.groupby('Event_Clock').apply(lambda x: x.index[-1])

    vol_df = df.groupby('Event_Clock')[vol_cols].sum()
    
    print(f"[*] Compressed into {len(sensorium):,} True Thermodynamic States.")

    print(f"[*] Constructing Phase-Space Density Maps (Volume at Price) for top nodes...")

    BINS = 100 
    LOOKBACK = 1440 # A rolling topographical map of historical liquidity
    
    target_symbols = symbols[:5] # Audit a subset to keep processing rapid
    
    passes = 0
    failures = 0
    yield_log = []
    
    for sym_idx, sym in enumerate(target_symbols):
        print(f"    -> Scanning Physical Terrain: [{sym}]")
        prices = sensorium[f"{sym}_M1_price"].values
        volumes = vol_df[f"{sym}_M1_vol"].values
        
        for i in range(LOOKBACK, len(sensorium) - 60, 20): # Step by 20 states
            
            hist_prices = prices[i-LOOKBACK:i]
            hist_vols = volumes[i-LOOKBACK:i]
            
            # 1. BUILD THE TOPOGRAPHY (Volume Profile)
            min_p, max_p = np.min(hist_prices), np.max(hist_prices)
            if max_p == min_p: continue
                
            bins = np.linspace(min_p, max_p, BINS)
            digitized = np.digitize(hist_prices, bins)
            
            vol_profile = np.zeros(BINS)
            for j in range(len(hist_prices)):
                if 0 <= digitized[j] < BINS:
                    vol_profile[digitized[j]] += hist_vols[j]
                    
            # 2. IDENTIFY THE WALLS AND THE VACUUMS
            active_bins = vol_profile[vol_profile > 0]
            if len(active_bins) < 10: continue
                
            hvn_threshold = np.percentile(active_bins, 80)
            lvn_threshold = np.percentile(active_bins, 20)
            
            curr_p = prices[i]
            curr_bin = np.digitize([curr_p], bins)[0]
            
            if not (5 < curr_bin < BINS - 5): continue
            
            # 3. THE PHYSICS TRIGGER: THE EDGE OF THE CLIFF
            # Condition: We are currently standing inside a Concrete Wall (HVN)
            if vol_profile[curr_bin] > hvn_threshold:
                
                # Check recent kinetic direction (Is it pushing against the edge?)
                recent_velocity = curr_p - prices[i-5]
                
                # Check UPWARD Void
                if recent_velocity > 0:
                    void_above = vol_profile[curr_bin+1:curr_bin+4]
                    if np.mean(void_above) < lvn_threshold:
                        
                        # 4. ACTION PHYSICS: Did price teleport through the vacuum?
                        future_prices = prices[i+1:i+61]
                        max_ext = np.max(future_prices)
                        max_draw = np.min(future_prices)
                        
                        # Gauge Inversion Fix: Prices are already log-transformed
                        release = abs(max_ext - curr_p) * 100
                        friction = abs(max_draw - curr_p) * 100
                        
                        # Falsification: Teleportation confirmed if release > 3x friction
                        if release > (friction * 3.0) and release > 0.05:
                            passes += 1
                            yield_log.append(release)
                        else:
                            failures += 1
                            
                # Check DOWNWARD Void
                elif recent_velocity < 0:
                    void_below = vol_profile[curr_bin-3:curr_bin]
                    if np.mean(void_below) < lvn_threshold:
                        
                        future_prices = prices[i+1:i+61]
                        max_ext = np.min(future_prices)
                        max_draw = np.max(future_prices)
                        
                        # Gauge Inversion Fix: Prices are already log-transformed
                        release = abs(max_ext - curr_p) * 100
                        friction = abs(max_draw - curr_p) * 100
                        
                        if release > (friction * 3.0) and release > 0.05:
                            passes += 1
                            yield_log.append(release)
                        else:
                            failures += 1

    total_events = passes + failures
    win_rate = (passes / total_events) * 100 if total_events > 0 else 0
    avg_yield = np.mean(yield_log) if yield_log else 0.0
    
    print("\n" + "="*80)
    print("AION TOPOGRAPHY MAPPER: THE CLIFF-EDGE AUDIT (6 MONTHS)")
    print("="*80)
    print(f"Total Structural Cliff-Edges Identified: {total_events}")
    print(f"Vacuum Teleportation Win Rate:           {win_rate:.2f}%")
    print(f"Average Kinetic Yield (Through Vacuum):  {avg_yield:.4f}%")
    print("="*80)
    
    if win_rate > 55:
        print("[ VERDICT ] PHYSICS CONFIRMED. PRICE TELEPORTS THROUGH LIQUIDITY VACUUMS.")
    else:
        print("[ VERDICT ] HYPOTHESIS FAILED. NOISE DETECTED. WALLS ABSORB ENERGY.")

if __name__ == "__main__":
    run_topography_mapper()
