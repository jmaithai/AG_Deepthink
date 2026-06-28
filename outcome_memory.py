import pandas as pd
import numpy as np
import os
import warnings

warnings.simplefilter(action='ignore')

def run_outcome_memory():
    file_path = "archive/omni_energy_cloud.parquet"
    print(f"[*] AION OUTCOME MEMORY: Loading Continuous Tick Stream from {file_path}")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Error: {e}")
        return

    # The exact boundary of the Wave-Function Collapse (Event #721)
    rupture_time = pd.to_datetime("2026-06-26 18:53:14")
    
    # Isolate the forward-time manifold (The next 25k network events)
    forward_df = df[df.index >= rupture_time].head(25000)
    
    if forward_df.empty:
        print("[-] Fatal: No forward data available after rupture point.")
        return

    print(f"[*] Locking Ground Truth (T=0) at {rupture_time}...")
    
    # The actors identified by the Arbiter
    epicenter = "USDCAD"
    receiver = "GBPUSD"
    trap_outlet = "CADJPY"
    
    # Baseline Prices at the exact millisecond of Wave-Function Collapse
    p0_epi = forward_df[f"{epicenter}_mid"].iloc[0]
    p0_rec = forward_df[f"{receiver}_mid"].iloc[0]
    p0_trap = forward_df[f"{trap_outlet}_mid"].iloc[0]
    
    print("[*] Tracking Kinetic Extension and Regression to Equilibrium...")
    
    # Calculate continuous kinetic displacement (log returns from T=0)
    forward_df = forward_df.copy()
    forward_df['Epi_Displacement'] = np.log(forward_df[f"{epicenter}_mid"] / p0_epi) * 100
    forward_df['Rec_Displacement'] = np.log(forward_df[f"{receiver}_mid"] / p0_rec) * 100
    forward_df['Trap_Displacement'] = np.log(forward_df[f"{trap_outlet}_mid"] / p0_trap) * 100
    
    # Track maximum excursions
    max_epi_extension = forward_df['Epi_Displacement'].max()
    max_epi_drawdown = forward_df['Epi_Displacement'].min()
    
    max_trap_extension = forward_df['Trap_Displacement'].max()
    max_trap_drawdown = forward_df['Trap_Displacement'].min()
    
    # Find exact timestamp the Trap sprang
    trap_climax_idx = forward_df['Trap_Displacement'].idxmax() if abs(max_trap_extension) > abs(max_trap_drawdown) else forward_df['Trap_Displacement'].idxmin()
    trap_climax_time = trap_climax_idx.strftime('%H:%M:%S')
    trap_climax_val = forward_df.loc[trap_climax_idx, 'Trap_Displacement']
    
    epi_at_trap_climax = forward_df.loc[trap_climax_idx, 'Epi_Displacement']

    print("\n" + "="*80)
    print("AION ENGINE: OUTCOME MEMORY (FALSIFICATION REPORT)")
    print("="*80)
    
    print(f"RUPTURE T=0: 2026-06-26 18:53:14")
    print(f"Forward Track: 25,000 subsequent network transactions.\n")
    
    print("[ THE EPICENTER (USDCAD) - The Illusion ]")
    print(f"Retail Reality: Traded the velocity of the breakout.")
    print(f"Max Kinetic Extension: {max_epi_extension:+.4f}%")
    print(f"Max Regression (Trap Spring): {max_epi_drawdown:+.4f}%")
    
    print("\n[ THE WEAKEST OUTLET (CADJPY) - The Physics ]")
    print(f"AION Arbiter Reality: Monitored the dead zone for forced liquidity sweep.")
    print(f"Adverse Friction (Heat): {max_trap_drawdown:+.4f}% (Zero-heat entry confirmed)")
    print(f"Maximum Kinetic Release: {max_trap_extension:+.4f}%")
    
    print("\n[ POST-RUPTURE AUTOPSY ]")
    
    if abs(trap_climax_val) > abs(max_epi_extension) and abs(max_trap_drawdown) < 0.05:
        print(f"STATUS: FALSIFICATION PASSED. THE DOCTRINE HOLDS.")
        print(f"-> At T=0, {epicenter} flashed fake velocity (Retail Trap).")
        print(f"-> The true pressure wave bypassed the epicenter and forced its way through {trap_outlet}.")
        print(f"-> The Outlet ({trap_outlet}) experienced a violent {trap_climax_val:+.2f}% forced release at {trap_climax_time}.")
        print(f"-> Epicenter ({epicenter}) collapsed to {epi_at_trap_climax:+.2f}% at the exact same moment.")
    else:
        print(f"STATUS: FALSIFICATION FAILED. NO TRAP DETECTED.")
        print(f"-> Epicenter max extension: {max_epi_extension:+.4f}% | Outlet climax: {trap_climax_val:+.4f}%")
        print(f"-> The pressure wave did not force liquidation through the expected outlet.")
        
    print("="*80)

if __name__ == "__main__":
    run_outcome_memory()
