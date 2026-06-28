import os
import sys
import sqlite3
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from information_ingestor import InformationIngestor
from manifold_solver import ManifoldSolver

def run_wkb_tunneling_backtest():
    db_path = "e:/spiderweb_core/AG_marketmap/roboforex/backend/regime.db"
    ingestor = InformationIngestor()
    
    trade_symbols = [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
        "XAUUSD", "XAGUSD", "XAUEUR"
    ]
    
    strand_map = {}
    for s in trade_symbols:
        strand_map[s] = ingestor.parse_symbol_nodes(s)
        
    print(f"[*] Initializing WKB Quantum Tunneling Engine...")
    
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
    
    print(f"[*] Loaded phase space matrix: {prices_df.shape[0]} bars x {prices_df.shape[1]} symbols.")
    
    # Solve pressures (Potential Energy Landscape)
    solver = ManifoldSolver(strand_map)
    pressures_df = solver.solve_pressures(prices_df)
    
    pressures_mean = pressures_df.mean()
    pressures_std = pressures_df.std()
    pressures_norm = (pressures_df - pressures_mean) / (pressures_std + 1e-12)
    
    # Proper Time (tau) Conversion
    # dtau = tick_volume / rolling_median(tick_volume)
    print("[*] Warping Spacetime to Proper Market Time (tau)...")
    vol_median = volume_df.rolling(48, min_periods=1).median().replace(0, 1)
    dtau_df = volume_df / vol_median
    
    # Velocity in proper time (v = dp / dtau)
    log_prices = np.log(prices_df)
    dp_df = log_prices.diff().fillna(0.0)
    velocity_tau = dp_df / (dtau_df + 1e-12)
    
    print("[*] Integrating Graph Wave-Diffusion (Topological Energy Flux)...")
    dtau_series = dtau_df.mean(axis=1) # Global proper time clock
    wave_res = solver.solve_graph_wave_diffusion(pressures_df, dtau_series, c2=0.5, gamma=0.1)
    wave_flux_df = wave_res["wave_flux"]
    
    print("[*] Solving WKB Tunneling Probabilities...")
    
    capital = 10000.0
    active_trades = []
    trade_log = []
    hbar = 1.0 # Market discrete minimum quantum (can be calibrated)
    
    # Pre-calculate strand tensors
    strand_q = {}
    strand_v = {}
    strand_mass = {}
    strand_vol = {}
    strand_F = {}
    
    for sym in trade_symbols:
        if sym in prices_df.columns:
            node_a, node_b = strand_map[sym]
            strand_q[sym] = pressures_norm[node_a] - (pressures_norm[node_b] if node_b else 0)
            if node_a in velocity_tau.columns and node_b in velocity_tau.columns:
                strand_v[sym] = velocity_tau[node_a] - velocity_tau[node_b]
            else:
                strand_v[sym] = velocity_tau[sym]
            
            # Pure instantaneous physics: Mass is determined by immediate order flow density (tick volume)
            # No lagging moving average variances.
            strand_mass[sym] = dtau_df[sym] + 1e-6
            strand_vol[sym] = dtau_df[sym] # Normalized volume serves as liquidity wall strength
            
            # Instantaneous Topological Wave Flux
            flux_a = wave_flux_df[f"{node_a}_wave_flux"] if f"{node_a}_wave_flux" in wave_flux_df.columns else pd.Series(0.0, index=prices_df.index)
            if node_b:
                flux_b = wave_flux_df[f"{node_b}_wave_flux"] if f"{node_b}_wave_flux" in wave_flux_df.columns else pd.Series(0.0, index=prices_df.index)
                strand_F[sym] = flux_a - flux_b
            else:
                strand_F[sym] = flux_a
            
    # Cross-sectional sizing mass
    mass_df = pd.DataFrame(strand_mass)
    global_avg_mass = mass_df.mean(axis=1)
    cross_sectional_scale = mass_df.div(global_avg_mass, axis=0)

    for i in range(50, len(prices_df)):
        t = prices_df.index[i]
        current_prices = prices_df.iloc[i]
        
        # 1. Evaluate Exit (Rupture or Equilibrium)
        for trade in active_trades[:]:
            sym = trade['symbol']
            entry_price = trade['entry_price']
            direction = trade['direction']
            
            q_current = strand_q[sym].iloc[i]
            
            # Exit if we cross zero (equilibrium)
            crossed_zero = (trade['entry_q'] > 0 and q_current <= 0) or (trade['entry_q'] < 0 and q_current >= 0)
            
            # Exit if WKB probability spikes (rupture detected mid-trade)
            # This cuts losers before they explode
            v_curr = strand_v[sym].iloc[i]
            m_curr = strand_mass[sym].iloc[i]
            vol_norm = strand_vol[sym].iloc[i]
            F_curr = strand_F[sym].iloc[i]
            
            E_k = 0.5 * m_curr * (v_curr ** 2)
            E_p = 0.5 * (q_current ** 2)
            V_0 = E_p * vol_norm
            
            # Apply Asymmetric Potential Tensor warp (Instantaneous Topological Wave Flux)
            alpha = 10.0 # Scale sensitivity to flux
            inner_product = np.clip(q_current * F_curr * alpha, -10.0, 10.0)
            V_asym = V_0 * np.exp(-inner_product)
            
            T = 1.0
            if E_k < V_asym:
                integral = np.sqrt(2 * m_curr * (V_asym - E_k)) * abs(q_current)
                T = np.exp(-2 * integral / hbar)
                
            ruptured = (T > 0.9)
            
            if crossed_zero or ruptured:
                exit_price = current_prices[sym]
                ret = (exit_price - entry_price) / entry_price if direction == 'LONG' else (entry_price - exit_price) / entry_price
                
                mass_scale = cross_sectional_scale.iloc[i][sym]
                if np.isnan(mass_scale): mass_scale = 1.0
                
                # Portfolio risk normalization (reduce leverage if many trades are open)
                portfolio_scale = 1.0 / max(1, len(active_trades))
                leverage = min(1.5, abs(trade['entry_q'])**2 * 5.0 * mass_scale) * portfolio_scale
                
                pnl_pct = ret * leverage
                capital_change = capital * pnl_pct
                capital += capital_change
                
                trade['exit_time'] = t
                trade['exit_price'] = exit_price
                trade['pnl_pct'] = pnl_pct
                trade['capital_change'] = capital_change
                trade['exit_reason'] = 'Equilibrium' if crossed_zero else 'Rupture_Bailout'
                trade_log.append(trade)
                active_trades.remove(trade)

        # 2. Evaluate Entry (Reflection)
        for sym in trade_symbols:
            if sym not in prices_df.columns: continue
            if any(t['symbol'] == sym for t in active_trades): continue
                
            q_current = strand_q[sym].iloc[i]
            if abs(q_current) < 0.5:
                continue # Spring is not stretched
                
            v_curr = strand_v[sym].iloc[i]
            m_curr = strand_mass[sym].iloc[i]
            vol_norm = strand_vol[sym].iloc[i]
            F_curr = strand_F[sym].iloc[i]
            
            # Incoming kinetic energy
            E_k = 0.5 * m_curr * (v_curr ** 2)
            # Potential energy of the stretch
            E_p = 0.5 * (q_current ** 2)
            # Base Potential barrier
            V_0 = E_p * vol_norm
            
            # Apply Asymmetric Potential Tensor warp
            alpha = 10.0
            inner_product = np.clip(q_current * F_curr * alpha, -10.0, 10.0)
            V_asym = V_0 * np.exp(-inner_product)
            
            # Physical Block: If the manifold is tilted so strongly that the barrier 
            # is degraded by >50%, reflection is physically impossible. Do not enter.
            if V_asym < (V_0 * 0.5):
                continue
            
            # WKB Tunneling Probability
            if E_k >= V_asym:
                T = 1.0 # Guaranteed breakthrough
            else:
                integral = np.sqrt(2 * m_curr * (V_asym - E_k)) * abs(q_current)
                T = np.exp(-2 * integral / hbar)
                
            # If reflection is nearly guaranteed (T < 0.1), enter reversion trade
            if T < 0.1:
                active_trades.append({
                    'symbol': sym,
                    'entry_time': t,
                    'entry_price': current_prices[sym],
                    'direction': 'SHORT' if q_current > 0 else 'LONG',
                    'entry_q': q_current,
                    'T': T
                })

    # Close open trades at the end
    for trade in active_trades:
        sym = trade['symbol']
        exit_price = prices_df.iloc[-1][sym]
        ret = (exit_price - trade['entry_price']) / trade['entry_price'] if trade['direction'] == 'LONG' else (trade['entry_price'] - exit_price) / trade['entry_price']
        
        mass_scale = cross_sectional_scale.iloc[-1][sym]
        if np.isnan(mass_scale): mass_scale = 1.0
        portfolio_scale = 1.0 / max(1, len(active_trades))
        leverage = min(1.5, abs(trade['entry_q'])**2 * 5.0 * mass_scale) * portfolio_scale
        
        pnl_pct = ret * leverage
        capital_change = capital * pnl_pct
        capital += capital_change
        trade['pnl_pct'] = pnl_pct
        trade['exit_reason'] = 'End_of_Data'
        trade_log.append(trade)

    print("\n" + "="*60)
    print("WKB QUANTUM TUNNELING ENGINE BACKTEST RESULTS")
    print("="*60)
    print(f"Starting Capital: $10,000.00")
    print(f"Ending Capital:   ${capital:,.2f}")
    
    total_ret = (capital - 10000.0) / 10000.0
    print(f"Total Return:     {total_ret:+.2%}")
    
    if len(trade_log) > 0:
        df_trades = pd.DataFrame(trade_log)
        wins = df_trades[df_trades['pnl_pct'] > 0]
        win_rate = len(wins) / len(df_trades)
        
        cum_ret = (1 + df_trades['pnl_pct']).cumprod()
        peak = cum_ret.cummax()
        dd = (cum_ret - peak) / peak
        
        mean_ret = df_trades['pnl_pct'].mean()
        std_ret = df_trades['pnl_pct'].std()
        sharpe = (mean_ret / std_ret) * np.sqrt(252 * 24 / (len(prices_df) / len(df_trades))) if std_ret > 0 else 0
        
        print(f"Max Drawdown:     {dd.min():.2%}")
        print(f"Win Rate:         {win_rate:.2%} ({len(wins)}/{len(df_trades)})")
        print(f"Total Trades:     {len(df_trades)}")
        print(f"Avg PnL per Trade:{mean_ret:+.2%}")
        print(f"Approx Sharpe:    {sharpe:.3f}")
        
        # Analyze bailout effectiveness
        bailouts = df_trades[df_trades['exit_reason'] == 'Rupture_Bailout']
        if len(bailouts) > 0:
            print(f"Rupture Bailouts: {len(bailouts)} trades cut early to prevent DD.")
    
    print("="*60)

if __name__ == "__main__":
    run_wkb_tunneling_backtest()
