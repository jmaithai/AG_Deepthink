import os
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

os.environ["PYTHONUTF8"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import physics_engine as pe
import kinematics_engine as ke
import policy_engine as po
import execution_engine as ee
import pandas as pd

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return RedirectResponse(url="/static/index.html")

@app.get("/api/snapshot")
def get_snapshot():
    pe.initialize_mt5()
    nodes, edges = pe.discover_topology()
    thermo_df = pe.fetch_and_warp_fluid(edges)
    pressure_df = pe.extract_hoberman_space(thermo_df, nodes, edges)
    v, m, q, J, dJ, S_norm, phi = ke.compute_kinematics(pressure_df)
    
    final_actions, current_phi = po.decide_active_inference(nodes, edges, v, m, q, J, dJ, phi)
    
    equity = 0.0
    account_info = pe.mt5.account_info()
    if account_info is not None:
        equity = account_info.equity
        
    pe.mt5.shutdown()
    
    # Format data for frontend
    action_data = []
    for i, e in enumerate(edges):
        action_data.append({
            "Symbol": e['symbol'],
            "U_risk": float(final_actions[i]),
            "Abs_U": abs(float(final_actions[i]))
        })
    action_data.sort(key=lambda x: x["Abs_U"], reverse=True)
    
    node_data = []
    P_last = pressure_df.iloc[-1].values
    v_last = v[-1].numpy()
    m_last = m[-1].numpy() / 1e6
    
    for i, node in enumerate(nodes):
        node_data.append({
            "node": node,
            "P": float(P_last[i]),
            "v": float(v_last[i]),
            "m": float(m_last[i])
        })
    
    return {
        "phi": float(current_phi),
        "entropy": float(S_norm[-1].item()),
        "active_nodes": len(nodes),
        "equity": equity,
        "actions": action_data,
        "nodes": node_data
    }

class ExecReq(BaseModel):
    dry_run: bool

@app.post("/api/execute")
def execute(req: ExecReq):
    pe.initialize_mt5()
    nodes, edges = pe.discover_topology()
    thermo_df = pe.fetch_and_warp_fluid(edges)
    pressure_df = pe.extract_hoberman_space(thermo_df, nodes, edges)
    v, m, q, J, dJ, S_norm, phi = ke.compute_kinematics(pressure_df)
    
    final_actions, current_phi = po.decide_active_inference(nodes, edges, v, m, q, J, dJ, phi)
    
    action_df = pd.DataFrame({'Symbol': [e['symbol'] for e in edges], 'U_risk': final_actions})
    action_df['Abs_U'] = action_df['U_risk'].abs()
    action_df = action_df.sort_values(by='Abs_U', ascending=False)
    
    df_manifest, total_risk = ee.execute_action_tensor(action_df, edges, current_phi, dry_run=req.dry_run)
    pe.mt5.shutdown()
    
    manifest_records = df_manifest.to_dict(orient="records") if not df_manifest.empty else []
    
    return {
        "manifest": manifest_records,
        "total_risk": float(total_risk)
    }
