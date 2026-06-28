import os
import sys
import sqlite3
import numpy as np
import pandas as pd

sys.path.append("e:/spiderweb_core/AG_marketmap/backend/experimental")
from information_ingestor import InformationIngestor
from manifold_solver import ManifoldSolver

db_path = "e:/spiderweb_core/AG_marketmap/roboforex/backend/regime.db"
ingestor = InformationIngestor()

trade_symbols = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
    "XAUUSD", "XAGUSD", "XAUEUR"
]

strand_map = {}
for s in trade_symbols:
    strand_map[s] = ingestor.parse_symbol_nodes(s)

conn = sqlite3.connect(db_path)
df_prices = pd.read_sql_query(
    "SELECT symbol, time, close FROM bar_history WHERE timeframe = 'H1' ORDER BY time, symbol",
    conn, parse_dates=["time"]
)
conn.close()

prices_df = df_prices.pivot(index="time", columns="symbol", values="close").ffill().bfill()
prices_df = prices_df[[s for s in trade_symbols if s in prices_df.columns]].copy()

solver = ManifoldSolver(strand_map)
pressures_df = solver.solve_pressures(prices_df)

pressures_mean = pressures_df.mean()
pressures_std = pressures_df.std()
pressures_norm = (pressures_df - pressures_mean) / pressures_std

log_prices = np.log(prices_df)
velocity = log_prices.diff().fillna(0.0)

aion_entropy = pd.Series(index=prices_df.index, dtype=float)

for i in range(50, len(prices_df)):
    window_vel = velocity.iloc[i-50:i].values
    norm_vel = (window_vel - np.mean(window_vel, axis=0)) / (np.std(window_vel, axis=0) + 1e-12)
    cov_matrix = np.cov(norm_vel.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    total = np.sum(np.abs(eigenvalues)) + 1e-12
    p_energy = np.abs(eigenvalues[idx]) / total
    p_energy = np.clip(p_energy, 1e-12, 1.0)
    entropy = -np.sum(p_energy * np.log2(p_energy))
    aion_entropy.iloc[i] = entropy

strand_q = {}
for sym in trade_symbols:
    if sym in prices_df.columns:
        node_a, node_b = strand_map[sym]
        strand_q[sym] = pressures_norm[node_a] - (pressures_norm[node_b] if node_b else 0)

# Also test cross-sectional mass scaling
mass_df = pd.DataFrame()
for sym in trade_symbols:
    if sym in velocity.columns:
        mass_df[sym] = 1.0 / (velocity[sym].rolling(48).var() + 1e-12)
global_avg_mass = mass_df.mean(axis=1)
cross_sectional_scale = mass_df.div(global_avg_mass, axis=0)

for threshold in [1.5, 2.0, 2.5, 3.0, 3.5]:
    capital = 10000.0
    active_trades = []
    trade_log = []
    
    for i in range(50, len(prices_df)):
        t = prices_df.index[i]
        current_prices = prices_df.iloc[i]
        
        for trade in active_trades[:]:
            sym = trade['symbol']
            entry_price = trade['entry_price']
            direction = trade['direction']
            
            q_current = strand_q[sym].iloc[i]
            crossed_zero = (trade['entry_q'] > 0 and q_current <= 0) or (trade['entry_q'] < 0 and q_current >= 0)
                
            if crossed_zero:
                exit_price = current_prices[sym]
                ret = (exit_price - entry_price) / entry_price if direction == 'LONG' else (entry_price - exit_price) / entry_price
                
                mass_scale = cross_sectional_scale.iloc[i][sym]
                if np.isnan(mass_scale): mass_scale = 1.0
                leverage = min(15.0, abs(trade['entry_q'])**2 * 5.0 * mass_scale)
                
                pnl_pct = ret * leverage
                capital_change = capital * pnl_pct
                capital += capital_change
                trade['pnl_pct'] = pnl_pct
                trade_log.append(trade)
                active_trades.remove(trade)
                
        s_val = aion_entropy.iloc[i]
        if s_val < threshold:
            for sym in trade_symbols:
                if sym not in prices_df.columns: continue
                if any(t['symbol'] == sym for t in active_trades): continue
                    
                q_current = strand_q[sym].iloc[i]
                if abs(q_current) > 0.5:
                    active_trades.append({
                        'symbol': sym,
                        'entry_price': current_prices[sym],
                        'direction': 'SHORT' if q_current > 0 else 'LONG',
                        'entry_q': q_current
                    })
                    
    # Close open trades
    for trade in active_trades:
        sym = trade['symbol']
        exit_price = prices_df.iloc[-1][sym]
        ret = (exit_price - trade['entry_price']) / trade['entry_price'] if trade['direction'] == 'LONG' else (trade['entry_price'] - exit_price) / trade['entry_price']
        
        mass_scale = cross_sectional_scale.iloc[-1][sym]
        if np.isnan(mass_scale): mass_scale = 1.0
        leverage = min(15.0, abs(trade['entry_q'])**2 * 5.0 * mass_scale)
        
        pnl_pct = ret * leverage
        capital_change = capital * pnl_pct
        capital += capital_change
        trade['pnl_pct'] = pnl_pct
        trade_log.append(trade)

    if len(trade_log) > 0:
        df = pd.DataFrame(trade_log)
        wins = df[df['pnl_pct'] > 0]
        win_rate = len(wins) / len(df)
        cum_ret = (1 + df['pnl_pct']).cumprod()
        dd = (cum_ret - cum_ret.cummax()) / cum_ret.cummax()
        print(f"Threshold {threshold}: Trades: {len(df)}, WR: {win_rate:.2%}, DD: {dd.min():.2%}, Ret: {(capital-10000)/10000:.2%}")

