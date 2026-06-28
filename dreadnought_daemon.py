import time
import os
import sys
import logging
import pandas as pd
from datetime import datetime

# Import engine modules
import physics_engine as pe
import kinematics_engine as ke
import policy_engine as po
import execution_engine as ee

# Force the log to exist exactly in your project folder
log_path = r'e:\spiderweb_core\AG_Deepthink\dreadnought.log'
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s - [DAEMON] - %(message)s'
)

# The Heartbeat
INTERVAL_SECONDS = 60 

# Safety Watchdog - ALWAYS default to True unless explicitly overridden
DRY_RUN = True

def run_daemon():
    logging.info("=" * 60)
    logging.info(f"[{datetime.now().strftime('%H:%M:%S')}] DREADNOUGHT DAEMON ONLINE")
    logging.info(f"[*] MODE: {'PAPER TRADE (DRY_RUN=True)' if DRY_RUN else 'LIVE TRADE (DRY_RUN=False)'}")
    logging.info("=" * 60)
    
    pe.initialize_mt5()
    
    try:
        while True:
            cycle_start = time.time()
            logging.info(f"[{datetime.now().strftime('%H:%M:%S')}] Initiating Manifold Scan...")
            
            try:
                # 1. Sense (Physics & Kinematics)
                nodes, edges = pe.discover_topology()
                thermo_df = pe.fetch_and_warp_fluid(edges)
                pressure_df = pe.extract_hoberman_space(thermo_df, nodes, edges)
                v, m, q, J, dJ, S_norm, phi = ke.compute_kinematics(pressure_df)
                
                # 2. Think (Policy)
                final_actions, current_phi = po.decide_active_inference(nodes, edges, v, m, q, J, dJ, phi)
                
                # 3. Format Tensor
                action_df = pd.DataFrame({'Symbol': [e['symbol'] for e in edges], 'U_risk': final_actions})
                action_df['Abs_U'] = action_df['U_risk'].abs()
                action_df = action_df.sort_values(by='Abs_U', ascending=False)
                
                # 4. Act (Execution)
                df_manifest, total_risk = ee.execute_action_tensor(action_df, edges, current_phi, dry_run=DRY_RUN)
                
                # Report
                logging.info(f"[*] Action Tensor Executed. Total Projected Risk: ${total_risk:.2f}")
                
            except Exception as e:
                logging.error(f"[!] Engine Error during cycle: {e}")
                
            # Heartbeat Wait
            elapsed = time.time() - cycle_start
            sleep_time = max(0, INTERVAL_SECONDS - elapsed)
            logging.info(f"[{datetime.now().strftime('%H:%M:%S')}] Heartbeat Complete. Sleeping for {sleep_time:.1f}s.")
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        logging.info("[SYSTEM] Daemon Shutdown requested via KeyboardInterrupt.")
    finally:
        pe.mt5.shutdown()
        logging.info("[SYSTEM] MT5 connection safely closed. Daemon offline.")

if __name__ == "__main__":
    run_daemon()
