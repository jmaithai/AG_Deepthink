import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
import pandas as pd
import numpy as np

# Suppress OpenMP collision and pandas warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.simplefilter(action='ignore', category=FutureWarning)

import physics_engine as pe
import kinematics_engine as ke

# =====================================================================
# LAYER 2: THE WORLD MODEL (LSTM MEMORY)
# =====================================================================
class HobermanWorldModel(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        # Inputs: q (tension) and J (flux) for all nodes = num_nodes * 2
        self.lstm = nn.LSTM(input_size=num_nodes * 2, hidden_size=64, num_layers=1, batch_first=True)
        # Outputs: predicted future q and future J = num_nodes * 2
        self.decoder = nn.Linear(64, num_nodes * 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        preds = self.decoder(out[:, -1, :]) # Predict next state based on sequence
        return preds

# =====================================================================
# LAYER 3: STATEFUL ACTIVE INFERENCE SGD OPTIMIZER
# =====================================================================
def decide_active_inference(active_nodes, active_edges, v, m, q, J, dJ, phi):
    print("[*] Igniting PyTorch World Model & Active Inference Engine...")
    
    N = len(active_nodes)
    E = len(active_edges)
    
    # 1. Prepare Thermodynamic Sequence Data for the LSTM
    # We combine tension (q) and flux (J) into a single feature tensor
    X = torch.cat([q, J], dim=1) # Shape: [Time, Nodes * 2]
    
    SEQ_LEN = 10
    if len(X) <= SEQ_LEN:
        SEQ_LEN = len(X) - 1
        
    X_seq, Y_target = [], []
    for i in range(len(X) - SEQ_LEN):
        X_seq.append(X[i : i + SEQ_LEN])
        Y_target.append(X[i + SEQ_LEN])
        
    if len(X_seq) > 0:
        X_seq = torch.stack(X_seq)
        Y_target = torch.stack(Y_target)
    
    # 2. Boot the LSTM World Model (Memory Recall)
    world_model = HobermanWorldModel(N)
    model_path = "world_model.pt"
    
    if os.path.exists(model_path):
        try:
            world_model.load_state_dict(torch.load(model_path, weights_only=True))
            print("[+] Previous LSTM Memory loaded. Engine remembers the wave topology.")
        except:
            print("[-] Memory corrupted. Initializing fresh World Model.")
    else:
        print("[*] No memory found. Initializing fresh World Model.")
        
    # 3. Online Adaptation (Train the World Model on the freshest waves)
    wm_optimizer = torch.optim.Adam(world_model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    
    if len(X_seq) > 0:
        print("[*] Training Predictive Horizon (50 Epochs)...")
        world_model.train()
        for epoch in range(50):
            wm_optimizer.zero_grad()
            preds = world_model(X_seq)
            loss = loss_fn(preds, Y_target)
            loss.backward()
            wm_optimizer.step()
            
        torch.save(world_model.state_dict(), model_path)
    
    # 4. Predict the Future State (q_traj, J_traj)
    world_model.eval()
    with torch.no_grad():
        if len(X) >= SEQ_LEN and SEQ_LEN > 0:
            latest_seq = X[-SEQ_LEN:].unsqueeze(0)
            future_pred = world_model(latest_seq).squeeze(0)
            q_pred = future_pred[:N]
            J_pred = future_pred[N:]
        else:
            # Fallback to current state if data is abnormally small
            q_pred = q[-1]
            J_pred = J[-1]
    
    # 5. Extract Present State Modifiers
    dJ_latest = dJ[-1].clone().detach()
    phi_latest = phi[-1].clone().detach()
    m_latest = m[-1].clone().detach()
    m_norm = m_latest / (torch.max(m_latest) + 1e-6)

    # 6. Reconstruct the Incidence Matrix (B)
    node_idx = {node: i for i, node in enumerate(active_nodes)}
    B_numpy = np.zeros((E, N))
    for i, edge in enumerate(active_edges):
        B_numpy[i, node_idx[edge['base']]] = 1.0
        B_numpy[i, node_idx[edge['quote']]] = -1.0
    B = torch.tensor(B_numpy, dtype=torch.float32)
    
    # 7. Portfolio Friction Memory (U_prev)
    U_prev = torch.zeros(E, dtype=torch.float32)
    action_path = "last_action.pt"
    if os.path.exists(action_path):
        try:
            U_prev = torch.load(action_path, weights_only=True)
            if len(U_prev) == E:
                print("[+] Previous Portfolio state loaded. Friction mechanics active.")
            else:
                U_prev = torch.zeros(E, dtype=torch.float32)
        except:
            pass

    # 8. PyTorch Active Inference Setup
    U_risk = torch.zeros(E, requires_grad=True, dtype=torch.float32)
    optimizer = torch.optim.Adam([U_risk], lr=0.01)
    
    # The Physics Laws
    LAMBDA_TSUNAMI = 100.0
    GAMMA_DECAY = 100.0
    BETA_BASE = 0.05
    ALPHA_DIV = 0.05
    FRICTION_COST = 0.05  # Penalty for turning over positions
    EPSILON = 1e-5
    dynamic_beta = BETA_BASE / (phi_latest + EPSILON)
    
    print("[*] Running PyTorch Free Energy Minimization (300 Epochs)...")
    
    for epoch in range(300):
        optimizer.zero_grad()
        
        node_exposure = torch.matmul(U_risk, B)
        
        # 1. Elastic Alignment (Targeting the PREDICTED future tension)
        L_elastic = phi_latest * torch.sum(node_exposure * (q_pred * 100.0))
        
        # 2. Kinetic Propulsion (Surfing the PREDICTED future wave)
        L_kinetic = -0.1 * torch.sum(node_exposure * J_pred)
        
        # 3. Tsunami Veto (On predicted alignment)
        wave_alignment = node_exposure * J_pred
        tsunami_penalty = LAMBDA_TSUNAMI * torch.sum(F_nn.relu(-wave_alignment) * (J_pred ** 2))
        
        # 4. Kinematic Decay
        decay_alignment = node_exposure * dJ_latest
        decay_penalty = GAMMA_DECAY * torch.sum(F_nn.relu(-decay_alignment) * torch.abs(dJ_latest))
        
        # 5. Volcanic Inertia
        L_inertia = dynamic_beta * torch.sum((node_exposure ** 2) * m_norm)
        
        # 6. Friction / Spread Cost (Memory of U_prev)
        # mathematically penalizes the optimizer if it changes positions for trivial alpha
        L_friction = FRICTION_COST * torch.sum(torch.abs(U_risk - U_prev))
        
        # 7. Diversification
        L_div = ALPHA_DIV * torch.sum(U_risk ** 2)
        
        F = L_elastic + L_kinetic + tsunami_penalty + decay_penalty + L_inertia + L_friction + L_div
        
        F.backward()
        optimizer.step()

    final_u = U_risk.detach()
    torch.save(final_u, action_path) # Save current actions for the next state's friction
    
    return final_u.numpy(), phi_latest.item()

def main():
    pe.initialize_mt5()
    active_nodes, active_edges = pe.discover_topology()
    if not active_nodes: return
        
    thermo_df = pe.fetch_and_warp_fluid(active_edges)
    pressure_df = pe.extract_hoberman_space(thermo_df, active_nodes, active_edges)
    pe.mt5.shutdown()
    
    v, m, q, J, dJ, S_norm, phi = ke.compute_kinematics(pressure_df)
    
    final_actions, current_phi = decide_active_inference(active_nodes, active_edges, v, m, q, J, dJ, phi)
    
    edge_names = [e['symbol'] for e in active_edges]
    action_df = pd.DataFrame({'Symbol': edge_names, 'U_risk': final_actions})
    action_df['Abs_U'] = action_df['U_risk'].abs()
    action_df = action_df[action_df['Abs_U'] > 0.05].sort_values(by='Abs_U', ascending=False)
    
    print("\n" + "="*80)
    print("ACTIVE INFERENCE PREDICTIVE ACTION TENSOR (TOP CONVICTION)")
    print("="*80)
    
    if current_phi < 0.5:
        print("[SYSTEM LOCKED] Criticality Detected (Volcano Erupting). 🔴")
    elif action_df.empty:
        print("[STANDBY] Laminar Flow active, but no edges possess sufficient kinematic conviction.")
    else:
        for idx, row in action_df.head(10).iterrows():
            direction = "BUY (LONG) " if row['U_risk'] > 0 else "SELL (SHORT)"
            print(f"{row['Symbol']:<12} | ACTION: {direction:<12} | CONVICTION (U_risk): {row['U_risk']:>8.4f}")
            
    print("="*80)

if __name__ == "__main__":
    main()
