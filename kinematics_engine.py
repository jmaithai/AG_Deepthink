import pandas as pd
import numpy as np
import torch
import warnings

# Suppress pandas FutureWarnings for clean console output
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import the titanium foundation from Module 1
import physics_engine as pe

# =====================================================================
# MODULE 2: PYTORCH KINEMATICS & THERMODYNAMICS
# =====================================================================

def compute_kinematics(pressure_df):
    print("[*] Igniting PyTorch Kinematics Engine...")
    
    # Convert Hoberman space to PyTorch Tensor [Time, Nodes]
    P = torch.tensor(pressure_df.values, dtype=torch.float32)
    N = P.shape[1]
    
    # 1. VELOCITY (v = dP/dTau)
    # The rate of pressure change per Proper Time tick
    v = torch.zeros_like(P)
    v[1:] = P[1:] - P[:-1]
    
    # 2. DYNAMIC MASS (m = 1 / variance of velocity)
    # Heavy nodes resist movement. Light nodes snap violently.
    v_df = pd.DataFrame(v.numpy())
    var_df = v_df.rolling(window=10).var().bfill().fillna(1e-6)
    variance = torch.tensor(var_df.values, dtype=torch.float32)
    m = 1.0 / (variance + 1e-6)
    
    # 3. KINETIC ENERGY (K = 0.5 * m * v^2)
    K = 0.5 * m * (v ** 2)
    
    # 4. TENSION (q = P - EMA)
    # Distance from the 21-state rolling thermodynamic equilibrium
    ema_df = pressure_df.ewm(span=21, adjust=False).mean()
    ema = torch.tensor(ema_df.values, dtype=torch.float32)
    q = P - ema
    
    # 5. TSUNAMI ENERGY FLUX (J = m * q * v)
    # The directional Poynting vector of macro power
    J = m * q * v
    
    # 6. KINEMATIC DECAY (dJ/dTau)
    # The Surfer's Exit signal. Acceleration of the macro power.
    dJ = torch.zeros_like(J)
    dJ[1:] = J[1:] - J[:-1]
    
    # 7. NETWORK ENTROPY (S)
    # Normalized Shannon Entropy across the manifold
    K_sum = torch.sum(K, dim=1, keepdim=True) + 1e-9
    p = K / K_sum
    S_raw = -torch.sum(p * torch.log(p + 1e-9), dim=1)
    
    # ln(N) is the theoretical maximum entropy for N nodes.
    # Normalizing bounds S strictly between 0.0 and 1.0.
    S_norm = S_raw / np.log(N)
    
    # 8. VOLCANIC PHASE GATE (Φ)
    # Sigmoid function that mathematically disables mean-reversion at criticality
    S_CRIT = 0.25  # Danger zone threshold (just above your 0.248 break point)
    K_STEEPNESS = 50.0
    phi = 1.0 / (1.0 + torch.exp(-K_STEEPNESS * (S_norm - S_CRIT)))
    
    return v, m, q, J, dJ, S_norm, phi

def main():
    # To run this standalone, it now relies purely on physics_engine logic.
    # It will attempt to connect to MT5 unless data_source is provided.
    try:
        pe.initialize_mt5()
        active_nodes, active_edges = pe.discover_topology()
        if not active_nodes:
            return
            
        thermo_df = pe.fetch_and_warp_fluid(active_edges)
        pressure_df = pe.extract_hoberman_space(thermo_df, active_nodes, active_edges)
        if hasattr(pe, 'mt5'):
            pe.mt5.shutdown()
    except Exception as e:
        print(f"[!] Could not run live test (MT5 missing or failed): {e}")
        return
    
    # 2. Compute PyTorch Kinematics
    v, m, q, J, dJ, S_norm, phi = compute_kinematics(pressure_df)
    
    # 3. Print Telemetry
    latest_idx = -1
    out_df = pd.DataFrame({
        'Pressure (P)': pressure_df.iloc[latest_idx].values,
        'Velocity (v)': v[latest_idx].numpy(),
        'Mass (m)': m[latest_idx].numpy(),
        'Tension (q)': q[latest_idx].numpy(),
        'J-Flux (Wave)': J[latest_idx].numpy(),
        'Decay (dJ/dτ)': dJ[latest_idx].numpy()
    }, index=active_nodes)
    
    print("\n" + "="*85)
    print("PYTORCH KINEMATIC TENSOR ARRAY (LATEST ENERGY STATE)")
    print("="*85)
    print(out_df.round(5).to_string())
    print("-" * 85)
    
    current_S = S_norm[latest_idx].item()
    current_phi = phi[latest_idx].item()
    
    print("\n" + "="*85)
    print("GLOBAL THERMODYNAMIC STATE")
    print("="*85)
    print(f"Network Entropy (S): {current_S:.6f} (Max = 1.0, Critical < {0.25})")
    print(f"Volcano Gate (Φ):    {current_phi:.6f} (1.0 = Laminar, 0.0 = Erupting)")
    print("="*85)
    
    if current_phi < 0.5:
        print("[WARNING] SYSTEM AT CRITICALITY. PYTORCH SGD WILL BE MATHEMATICALLY LOCKED. 🔴")
    else:
        print("[CLEAR] LAMINAR FLOW DETECTED. ELASTIC MEAN-REVERSION IS ACTIVE. 🟢")

if __name__ == "__main__":
    main()
