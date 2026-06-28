import pandas as pd
import numpy as np
import os
import warnings
import torch

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import physics_engine as pe
import kinematics_engine as ke

def extract_frequency_physics(df, active_nodes, active_edges, tf_label):
    """Isolates a specific timeframe and inverts its Hoberman Space."""
    cols = [f"{e['symbol']}_{tf_label}_price" for e in active_edges]
    sub_df = df[cols].copy()
    sub_df.columns = [c.replace(f"_{tf_label}_price", "_price") for c in sub_df.columns]
    
    P = pe.extract_hoberman_space(sub_df, active_nodes, active_edges)
    return ke.compute_kinematics(P)

def run_resonance_mapper():
    file_path = "archive/fractal_manifold_6M.parquet"
    print(f"[*] Loading Fractal Tensor: {file_path}")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Critical Error: Data missing. {e}")
        return
        
    print("[*] Warping Chronological Time into Thermodynamic States (\u03C4)...")
    vol_cols = [col for col in df.columns if col.endswith('_M1_vol')]
    df['network_kinetic_energy'] = df[vol_cols].sum(axis=1)
    df['cumulative_energy'] = df['network_kinetic_energy'].cumsum()
    df['Energy_State'] = (df['cumulative_energy'] // pe.ENERGY_THRESHOLD).astype(int)
    
    # We keep the raw timestamp so we can map the exact chronological moment the physics triggered
    df['chronological_time'] = df.index
    thermo_df = df.groupby('Energy_State').last()
    
    print(f"[+] Extracted {len(thermo_df)} Fractal Proper Time states.")
    
    active_edges = []
    active_nodes = set()
    price_cols = [col for col in df.columns if col.endswith('_M1_price')]
    for col in price_cols:
        sym = col.replace('_M1_price', '')
        base, quote = sym[:3], sym[3:6]
        active_edges.append({'symbol': sym, 'base': base, 'quote': quote})
        active_nodes.update([base, quote])
        
    active_nodes = sorted(list(active_nodes))
    
    print("[*] Computing Multi-Dimensional Physics (H1, M15, M1)...")
    v_H1, m_H1, q_H1, J_H1, dJ_H1, S_H1, phi_H1 = extract_frequency_physics(thermo_df, active_nodes, active_edges, "H1")
    v_M15, m_M15, q_M15, J_M15, dJ_M15, S_M15, phi_M15 = extract_frequency_physics(thermo_df, active_nodes, active_edges, "M15")
    v_M1, m_M1, q_M1, J_M1, dJ_M1, S_M1, phi_M1 = extract_frequency_physics(thermo_df, active_nodes, active_edges, "M1")
    
    print("[*] Scanning Timeline for Thermodynamic Resonance...")
    
    singularity_events = []
    
    # We map the 30 edges directly to find localized alignment
    for edge in active_edges:
        sym = edge['symbol']
        base, quote = edge['base'], edge['quote']
        base_idx = active_nodes.index(base)
        quote_idx = active_nodes.index(quote)
        
        # 1. MACRO POTENTIAL: Tension differential (H1)
        # If this is high, the institutional spring is heavily wound.
        tension_H1 = q_H1[:, base_idx] - q_H1[:, quote_idx]
        
        # 2. MESO GRADIENT: Flux differential (M15)
        # The transition current begins to move.
        flux_M15 = J_M15[:, base_idx] - J_M15[:, quote_idx]
        
        # 3. MICRO KINETIC: Flux differential (M1)
        # The immediate trigger wave.
        flux_M1 = J_M1[:, base_idx] - J_M1[:, quote_idx]
        
        # RESONANCE = Absolute magnitude of Macro Tension * Meso Flux * Micro Flux
        # Resonance is massive ONLY when all three dimensions align and compound
        resonance_score = torch.abs(tension_H1 * flux_M15 * flux_M1)
        
        for i in range(150, len(thermo_df)):
            # We only record states where the micro wave is not in total chaos (phi_M1 > 0.5)
            if phi_M1[i].item() > 0.5:
                
                t_h1 = tension_H1[i].item()
                f_m15 = flux_M15[i].item()
                f_m1 = flux_M1[i].item()
                r_score = resonance_score[i].item()
                
                if r_score > 1e-6: # Baseline filter to drop the noise floor
                    event_time = thermo_df.iloc[i]['chronological_time']
                    singularity_events.append({
                        'Energy_State': i,
                        'Chronological_Time': event_time,
                        'Symbol': sym,
                        'Vector': 'LONG' if f_m1 > 0 else 'SHORT',
                        'Macro_Tension_H1': t_h1,
                        'Meso_Flux_M15': f_m15,
                        'Micro_Flux_M1': f_m1,
                        'Resonance_Score': r_score
                    })

    print("[*] Processing Rupture Telemetry...")
    events_df = pd.DataFrame(singularity_events)
    if events_df.empty:
        print("[-] No Singularity Events found in the 6-month manifold.")
        return
        
    events_df = events_df.sort_values(by='Resonance_Score', ascending=False)
    
    # Filter out clustered events (same symbol within a few hours) to get distinct macro events
    events_df['Time_Group'] = events_df['Chronological_Time'].dt.floor('D')
    distinct_events = events_df.drop_duplicates(subset=['Symbol', 'Time_Group']).head(20)
    
    # Formatting for clean output
    distinct_events = distinct_events.copy()
    distinct_events['Macro_Tension_H1'] = distinct_events['Macro_Tension_H1'].apply(lambda x: f"{x:+.5f}")
    distinct_events['Meso_Flux_M15'] = distinct_events['Meso_Flux_M15'].apply(lambda x: f"{x:+.5f}")
    distinct_events['Micro_Flux_M1'] = distinct_events['Micro_Flux_M1'].apply(lambda x: f"{x:+.5f}")
    distinct_events['Resonance_Score'] = distinct_events['Resonance_Score'].apply(lambda x: f"{x:.8f}")
    
    print("\n" + "="*110)
    print("FRACTAL SINGULARITY POINTS (TOP HARMONIC RUPTURES - 6 MONTHS)")
    print("="*110)
    display_cols = ['Chronological_Time', 'Symbol', 'Vector', 'Macro_Tension_H1', 'Meso_Flux_M15', 'Micro_Flux_M1', 'Resonance_Score']
    print(distinct_events[display_cols].to_string(index=False))
    print("="*110)
    print(f"Total Structural Ruptures Detected: {len(events_df)}")

if __name__ == "__main__":
    run_resonance_mapper()
