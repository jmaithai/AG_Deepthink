import streamlit as st
import time
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch

# Ensure UTF-8 encoding and suppress OpenMP collision for all child processes
os.environ["PYTHONUTF8"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import our engine modules
import physics_engine as pe
import kinematics_engine as ke
import policy_engine as po
import execution_engine as ee

st.set_page_config(page_title="Dreadnought Command Bridge", layout="wide")

# Custom CSS for Dark Cyberpunk Theme
st.markdown("""
<style>
    /* Dark space background */
    .stApp {
        background-color: #0a0a0f;
        color: #e0e0e0;
        font-family: 'Courier New', Courier, monospace;
    }
    /* Headers and Text */
    h1, h2, h3 {
        color: #00f0ff !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(0, 240, 255, 0.5);
    }
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #ffd700 !important;
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 10, 15, 0.9) !important;
        border-right: 1px solid rgba(0, 240, 255, 0.3);
    }
    /* Buttons */
    .stButton > button {
        background-color: transparent !important;
        color: #00f0ff !important;
        border: 1px solid #00f0ff !important;
        box-shadow: 0 0 10px rgba(0, 240, 255, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: rgba(0, 240, 255, 0.1) !important;
        box-shadow: 0 0 20px rgba(0, 240, 255, 0.6);
    }
    /* Execute Button (Primary) */
    .stButton > button[data-testid="baseButton-primary"] {
        color: #ff003c !important;
        border: 1px solid #ff003c !important;
        box-shadow: 0 0 10px rgba(255, 0, 60, 0.3);
    }
    .stButton > button[data-testid="baseButton-primary"]:hover {
        background-color: rgba(255, 0, 60, 0.1) !important;
        box-shadow: 0 0 20px rgba(255, 0, 60, 0.6);
    }
</style>
""", unsafe_allow_html=True)

st.title("🌌 Dreadnought Command Bridge")
st.markdown("---")

@st.cache_data(ttl=60)
def get_snapshot():
    """Gathers the latest manifold state and caches it for 60s."""
    pe.initialize_mt5()
    nodes, edges = pe.discover_topology()
    thermo_df = pe.fetch_and_warp_fluid(edges)
    pressure_df = pe.extract_hoberman_space(thermo_df, nodes, edges)
    v, m, q, J, dJ, S_norm, phi = ke.compute_kinematics(pressure_df)
    
    # Get Conviction Tensor
    final_actions, current_phi = po.decide_active_inference(nodes, edges, v, m, q, J, dJ, phi)
    
    # Get Account Equity
    equity = 0.0
    account_info = pe.mt5.account_info()
    if account_info is not None:
        equity = account_info.equity
        
    pe.mt5.shutdown()
    
    return nodes, edges, pressure_df, v, m, q, J, dJ, phi, final_actions, current_phi, equity

# --- UI LOGIC ---
try:
    nodes, edges, P, v, m, q, J, dJ, S, final_actions, phi, equity = get_snapshot()
    
    # Sidebar: Vital Signs & Controls
    st.sidebar.header("🛡️ Vital Signs")
    st.sidebar.metric("Volcano Gate (Φ)", f"{phi:.4f}")
    st.sidebar.metric("Entropy (S)", f"{S[-1].item():.4f}")
    st.sidebar.metric("Active Nodes", len(nodes))
    st.sidebar.markdown("---")
    
    if st.sidebar.button("Force Snapshot Refresh"):
        st.cache_data.clear()
        st.rerun()
        
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Execution Mode")
    exec_mode = st.sidebar.radio("Mode", ["Paper Trade (DRY RUN)", "🔴 LIVE TRADE"], index=0)
    dry_run = exec_mode == "Paper Trade (DRY RUN)"
    
    # Main Area Top: Equity & Execution
    col1, col2 = st.columns([2, 1])
    with col1:
        st.metric("Account Equity", f"${equity:,.2f}")
    with col2:
        if st.button("🚀 EXECUTE ACTION TENSOR", type="primary", use_container_width=True):
            with st.spinner("Executing..."):
                pe.initialize_mt5()
                action_df = pd.DataFrame({'Symbol': [e['symbol'] for e in edges], 'U_risk': final_actions})
                action_df['Abs_U'] = action_df['U_risk'].abs()
                action_df = action_df.sort_values(by='Abs_U', ascending=False)
                
                df_manifest, total_risk = ee.execute_action_tensor(action_df, edges, phi, dry_run=dry_run)
                pe.mt5.shutdown()
                
                st.session_state['last_manifest'] = df_manifest
                st.session_state['last_risk'] = total_risk
                st.success("Execution Complete.")

    if 'last_manifest' in st.session_state and st.session_state['last_manifest'] is not None:
        st.header("📋 Execution Manifest")
        # Color coding for manifest actions
        def color_manifest_action(val):
            if val == "BUY":
                return 'color: #00ff41; font-weight: bold;'
            elif val == "SELL":
                return 'color: #ff003c; font-weight: bold;'
            elif val == "VETO":
                return 'color: #ffd700; font-weight: bold;'
            return ''
        
        styled_manifest = st.session_state['last_manifest'].style.map(color_manifest_action, subset=['ACTION'])
        st.dataframe(styled_manifest, width='stretch')
        st.info(f"Total Projected Risk (2.5 ATR Stop): ${st.session_state['last_risk']:,.2f}")
        st.markdown("---")

    # Color coding function for the dataframe
    def color_action(val):
        color = '#00ff41' if val > 0 else '#ff003c'
        return f'color: {color}; font-weight: bold;'

    # Main: Active Inference Terminal
    st.header("🎯 Active Inference Tensor")
    action_df = pd.DataFrame({'Symbol': [e['symbol'] for e in edges], 'U_risk': final_actions})
    action_df['Abs_U'] = action_df['U_risk'].abs()
    action_df = action_df.sort_values(by='Abs_U', ascending=False)
    
    # Style the dataframe
    styled_df = action_df[['Symbol', 'U_risk', 'Abs_U']].head(10).style.map(color_action, subset=['U_risk'])
    st.dataframe(styled_df, width='stretch')
    
    # Manifold Visualization (3D Scatter of Pressure)
    st.header("🌐 Manifold Geometry (P-Space)")
    # Simple mapping of Node Pressures to 3D for visualization
    coords = pd.DataFrame(index=nodes)
    coords['x'] = P.iloc[-1].values
    coords['y'] = v[-1].numpy()
    coords['z'] = m[-1].numpy() / 1e6 # Scale mass
    
    fig = px.scatter_3d(
        coords, x='x', y='y', z='z', 
        color=coords.index, size_max=10,
        color_discrete_sequence=px.colors.qualitative.Alphabet
    )
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis=dict(gridcolor='#1a1a2e', zerolinecolor='#00f0ff'),
            yaxis=dict(gridcolor='#1a1a2e', zerolinecolor='#00f0ff'),
            zaxis=dict(gridcolor='#1a1a2e', zerolinecolor='#00f0ff'),
        )
    )
    st.plotly_chart(fig, width='stretch')

except Exception as e:
    st.error(f"Engine Offline: {e}")
    if st.button("Retry Connection"):
        st.rerun()
