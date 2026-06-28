import os
import sys
import sqlite3
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from wkb_tunneling_engine import run_wkb_tunneling_backtest
from information_ingestor import InformationIngestor
from manifold_solver import ManifoldSolver

db_path = "e:/spiderweb_core/AG_marketmap/roboforex/backend/regime.db"
ingestor = InformationIngestor()

trade_symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "XAUUSD", "XAGUSD", "XAUEUR"]
strand_map = {s: ingestor.parse_symbol_nodes(s) for s in trade_symbols}

conn = sqlite3.connect(db_path)
df_history = pd.read_sql_query(
    "SELECT symbol, time, close, tick_volume FROM bar_history WHERE timeframe = 'H1' ORDER BY time, symbol",
    conn, parse_dates=["time"]
)
conn.close()

prices_df = df_history.pivot(index="time", columns="symbol", values="close").ffill().bfill()
prices_df = prices_df[[s for s in trade_symbols if s in prices_df.columns]].copy()

volume_df = df_history.pivot(index="time", columns="symbol", values="tick_volume").ffill().bfill()
volume_df = volume_df[[s for s in trade_symbols if s in volume_df.columns]].copy()

solver = ManifoldSolver(strand_map)
pressures_df = solver.solve_pressures(prices_df)

pressures_mean = pressures_df.mean()
pressures_std = pressures_df.std()
pressures_norm = (pressures_df - pressures_mean) / (pressures_std + 1e-12)

vol_median = volume_df.rolling(48, min_periods=1).median().replace(0, 1)
dtau_df = volume_df / vol_median

log_prices = np.log(prices_df)
dp_df = log_prices.diff().fillna(0.0)
velocity_tau = dp_df / (dtau_df + 1e-12)

capital = 10000.0
active_trades = []
trade_log = []
hbar = 1.0

strand_q = {}
strand_v = {}
strand_mass = {}
strand_vol = {}

for sym in trade_symbols:
    if sym in prices_df.columns:
        node_a, node_b = strand_map[sym]
        strand_q[sym] = pressures_norm[node_a] - (pressures_norm[node_b] if node_b else 0)
        if node_a in velocity_tau.columns and node_b in velocity_tau.columns:
            strand_v[sym] = velocity_tau[node_a] - velocity_tau[node_b]
        else:
            strand_v[sym] = velocity_tau[sym]
        strand_mass[sym] = dtau_df[sym] + 1e-6
        strand_vol[sym] = dtau_df[sym]
        
    print("[*] Integrating Graph Wave-Diffusion...")
    dtau_series = dtau_df.mean(axis=1)
    wave_res = solver.solve_graph_wave_diffusion(pressures_df, dtau_series, c2=0.5, gamma=0.1)
    wave_flux_df = wave_res["wave_flux"]
    
    strand_F = {}
    for sym in trade_symbols:
        node_a, node_b = strand_map[sym]
        flux_a = wave_flux_df[f"{node_a}_wave_flux"] if f"{node_a}_wave_flux" in wave_flux_df.columns else pd.Series(0.0, index=prices_df.index)
        if node_b:
            flux_b = wave_flux_df[f"{node_b}_wave_flux"] if f"{node_b}_wave_flux" in wave_flux_df.columns else pd.Series(0.0, index=prices_df.index)
            strand_F[sym] = flux_a - flux_b
        else:
            strand_F[sym] = flux_a

mass_df = pd.DataFrame(strand_mass)
global_avg_mass = mass_df.mean(axis=1)
cross_sectional_scale = mass_df.div(global_avg_mass, axis=0)

capital_curve = []
for i in range(50, len(prices_df)):
    t = prices_df.index[i]
    current_prices = prices_df.iloc[i]
    
    for trade in active_trades[:]:
        sym = trade['symbol']
        q_current = strand_q[sym].iloc[i]
        crossed_zero = (trade['entry_q'] > 0 and q_current <= 0) or (trade['entry_q'] < 0 and q_current >= 0)
        
        v_curr = strand_v[sym].iloc[i]
        m_curr = strand_mass[sym].iloc[i]
        vol_norm = strand_vol[sym].iloc[i]
        
        E_k = 0.5 * m_curr * (v_curr ** 2)
        E_p = 0.5 * (q_current ** 2)
        V_0 = E_p * vol_norm
        
        T = 1.0
        if E_k < V_0:
            integral = np.sqrt(2 * m_curr * (V_0 - E_k)) * abs(q_current)
            T = np.exp(-2 * integral / hbar)
            
        ruptured = (T > 0.9)
        
        if crossed_zero or ruptured:
            exit_price = current_prices[sym]
            ret = (exit_price - trade['entry_price']) / trade['entry_price'] if trade['direction'] == 'LONG' else (trade['entry_price'] - exit_price) / trade['entry_price']
            
            mass_scale = cross_sectional_scale.iloc[i][sym]
            if np.isnan(mass_scale): mass_scale = 1.0
            portfolio_scale = 1.0 / max(1, len(active_trades))
            leverage = min(5.0, abs(trade['entry_q'])**2 * 5.0 * mass_scale) * portfolio_scale
            
            pnl_pct = ret * leverage
            capital_change = capital * pnl_pct
            capital += capital_change
            
            trade['exit_time'] = t
            trade['exit_price'] = exit_price
            trade['pnl_pct'] = pnl_pct
            trade['capital_change'] = capital_change
            trade['exit_reason'] = 'Equilibrium' if crossed_zero else 'Rupture_Bailout'
            trade['exit_T'] = T
            trade['bars_held'] = i - trade['entry_idx']
            trade_log.append(trade)
            active_trades.remove(trade)

    for sym in trade_symbols:
        if sym not in prices_df.columns: continue
        if any(t['symbol'] == sym for t in active_trades): continue
            
        q_current = strand_q[sym].iloc[i]
        if abs(q_current) < 0.5: continue
            
        v_curr = strand_v[sym].iloc[i]
        m_curr = strand_mass[sym].iloc[i]
        vol_norm = strand_vol[sym].iloc[i]
        
        E_k = 0.5 * m_curr * (v_curr ** 2)
        E_p = 0.5 * (q_current ** 2)
        V_0 = E_p * vol_norm
        
        F_curr = strand_F[sym].iloc[i]
        
        alpha = 10.0
        inner_product = np.clip(q_current * F_curr * alpha, -10.0, 10.0)
        V_asym = V_0 * np.exp(-inner_product)
        
        if E_k >= V_asym: T = 1.0
        else:
            integral = np.sqrt(2 * m_curr * (V_asym - E_k)) * abs(q_current)
            T = np.exp(-2 * integral / hbar)
            
        if T < 0.1 and V_asym >= (V_0 * 0.5):
            active_trades.append({
                'symbol': sym,
                'entry_time': t,
                'entry_price': current_prices[sym],
                'direction': 'SHORT' if q_current > 0 else 'LONG',
                'entry_q': q_current,
                'entry_T': T,
                'entry_F': F_curr,
                'entry_idx': i
            })
            
    capital_curve.append({'time': t, 'capital': capital})

df_trades = pd.DataFrame(trade_log)
df_trades.sort_values(by='pnl_pct', ascending=True, inplace=True)

print(f"Total Trades: {len(df_trades)}")
print(f"Total Losers: {len(df_trades[df_trades['pnl_pct'] < 0])}")

print("\n--- FORENSIC AUDIT: 5 WORST TRADES ---")
worst_trades = df_trades.head(5)
for idx, row in worst_trades.iterrows():
    print(f"\nSymbol: {row['symbol']} | Direction: {row['direction']}")
    print(f"Entry: {row['entry_time']} | Exit: {row['exit_time']} | Held: {row['bars_held']} bars")
    print(f"Entry Tension (q): {row['entry_q']:.4f} | Entry Prob (T): {row['entry_T']:.4f} | Entry F: {row['entry_F']:.4f}")
    print(f"Exit Prob (T): {row['exit_T']:.4f} | Reason: {row['exit_reason']}")
    print(f"PnL: {row['pnl_pct']:.2%} | Capital Impact: ${row['capital_change']:.2f}")

print("\n--- MACRO SHOCK ANALYSIS ---")
df_losers = df_trades[df_trades['pnl_pct'] < 0]
exit_counts = df_losers.groupby('exit_time').size()
worst_exits = exit_counts[exit_counts > 1].sort_values(ascending=False).head(5)
print("Times with massive simultaneous stop-outs (Correlated Ruptures):")
print(worst_exits)
