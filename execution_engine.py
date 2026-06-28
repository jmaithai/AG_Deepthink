import os
import warnings
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

# Suppress OpenMP collision and pandas warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.simplefilter(action='ignore', category=FutureWarning)

import physics_engine as pe
import kinematics_engine as ke
import policy_engine as po

# =====================================================================
# MODULE 4: EXECUTION & COMPOUNDING (THE ASSASSIN'S HAND)
# =====================================================================

TARGET_RISK_PCT = 0.15      # 15% Portfolio Risk Threshold
CONVICTION_THRESHOLD = 0.00 # Minimum absolute U_risk to execute

MAX_POSITIONS = 5           # Top 5 Conviction Trades
DRY_RUN = True              # SAFETY SWITCH: Do not change yet.

def get_atr(symbol, timeframe=mt5.TIMEFRAME_H1, period=14):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, period + 1)
    if rates is None or len(rates) < period:
        return 0.001
    df = pd.DataFrame(rates)
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    return df['tr'].mean()

def execute_action_tensor(action_df, active_edges, current_phi, dry_run=True):
    print("[*] Engaging Execution & Compounding Protocol...")
    
    if current_phi < 0.5:
        print("[WARNING] Volcanic Gate Closed (Φ < 0.5). Forcing system to STAND ASIDE. 🔴")
        return

    account_info = mt5.account_info()
    if account_info is None:
        print("[-] Failed to retrieve MT5 account info. Ensure terminal is open.")
        return
        
    equity = account_info.equity
    currency = account_info.currency
    print(f"[+] Current MT5 Account Equity: {equity:,.2f} {currency}")
    
    valid_signals = action_df[action_df['Abs_U'] >= CONVICTION_THRESHOLD].head(MAX_POSITIONS)
    
    if valid_signals.empty:
        print("[-] No edges breached the conviction threshold. Holding state.")
        return

    total_risk_dollars = equity * TARGET_RISK_PCT
    total_conviction = valid_signals['Abs_U'].sum()

    manifest = []
    accumulated_risk = 0.0

    for _, row in valid_signals.iterrows():
        sym = row['Symbol']
        u_risk = row['U_risk']
        
        sym_info = mt5.symbol_info(sym)
        if sym_info is None or not sym_info.visible:
            continue
            
        atr = get_atr(sym)
        if atr <= 0: atr = 0.001
        
        # 1. Base proportional risk
        weight = row['Abs_U'] / total_conviction
        base_trade_risk = total_risk_dollars * weight
        
        # 2. Spectral Kelly + Log-Concave Compounding
        # Smooths outlier blowups (prevents massive ruin on extreme PyTorch signals)
        spectral_multiplier = np.log1p(row['Abs_U']) 
        compounded_risk = base_trade_risk * spectral_multiplier
        
        # 3. Volatility Sizing (2.5 ATR Hard Stop)
        sl_distance = atr * 2.5
        
        tick_value = sym_info.trade_tick_value
        tick_size = sym_info.trade_tick_size
        
        if tick_size == 0 or tick_value == 0: continue
        
        loss_per_lot = (sl_distance / tick_size) * tick_value
        if loss_per_lot == 0: continue
        
        raw_lot = compounded_risk / loss_per_lot
        
        # MT5 Broker rounding constraints
        lot = round(raw_lot / sym_info.volume_step) * sym_info.volume_step
        lot = max(sym_info.volume_min, min(lot, sym_info.volume_max))
        lot = round(lot, 2)
        
        actual_trade_risk = lot * loss_per_lot
        
        # --- THE ARCHITECT'S CAPITAL VETO ---
        # 1. If the broker's minimum physical lot requires more risk than our ENTIRE 15% budget, VETO.
        if actual_trade_risk > total_risk_dollars:
            manifest.append({
                "SYMBOL": sym,
                "ACTION": "VETO",
                "CONVICTION": round(u_risk, 4),
                "ATR": round(atr, 5),
                "RISK ($)": 0.00,
                "LOTS": 0.00,
                "STATUS": f"VETO: Min physical risk ${actual_trade_risk:.2f} > Max Budget"
            })
            continue
            
        # 2. Secondary Veto: Don't exceed remaining total portfolio budget (with small slippage buffer)
        if accumulated_risk + actual_trade_risk > total_risk_dollars * 1.1: 
             manifest.append({
                "SYMBOL": sym,
                "ACTION": "VETO",
                "CONVICTION": round(u_risk, 4),
                "ATR": round(atr, 5),
                "RISK ($)": 0.00,
                "LOTS": 0.00,
                "STATUS": f"VETO: Portfolio Risk Cap Reached"
            })
             continue
        
        action_type = mt5.ORDER_TYPE_BUY if u_risk > 0 else mt5.ORDER_TYPE_SELL
        action_str = "BUY" if u_risk > 0 else "SELL"
        
        if dry_run:
            status = "DRY_RUN (SAFE)"
        else:
            price = mt5.symbol_info_tick(sym).ask if u_risk > 0 else mt5.symbol_info_tick(sym).bid
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": sym,
                "volume": float(lot),
                "type": action_type,
                "price": price,
                "deviation": 20,
                "magic": 777777,
                "comment": "Macro_Physics",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                status = f"FAILED: {result.comment}"
            else:
                status = f"FIRED (Tk #{result.order})"
        
        accumulated_risk += actual_trade_risk
                
        manifest.append({
            "SYMBOL": sym,
            "ACTION": action_str,
            "CONVICTION": round(u_risk, 4),
            "ATR": round(atr, 5),
            "RISK ($)": round(actual_trade_risk, 2),
            "LOTS": lot,
            "STATUS": status
        })

    print("\n" + "="*110)
    print("FINAL EXECUTION MANIFEST (LOT SIZES)")
    print("="*110)
    df_manifest = pd.DataFrame(manifest)
    total_projected_risk = 0.0
    if not df_manifest.empty:
        print(df_manifest.to_string(index=False))
        total_projected_risk = df_manifest['RISK ($)'].sum()
        print("-" * 110)
        print(f"TOTAL PROJECTED RISK (2.5 ATR STOP): ${total_projected_risk:,.2f} ({(total_projected_risk/equity)*100:.2f}% of Equity)")
    else:
        print("[-] No viable orders calculated.")
    print("="*110)
    return df_manifest, total_projected_risk

def main():
    pe.initialize_mt5()
    active_nodes, active_edges = pe.discover_topology()
    if not active_nodes: return
        
    thermo_df = pe.fetch_and_warp_fluid(active_edges)
    pressure_df = pe.extract_hoberman_space(thermo_df, active_nodes, active_edges)
    
    v, m, q, J, dJ, S_norm, phi = ke.compute_kinematics(pressure_df)
    
    # We call policy_engine directly
    final_actions, current_phi = po.decide_active_inference(active_nodes, active_edges, v, m, q, J, dJ, phi)
    
    edge_names = [e['symbol'] for e in active_edges]
    action_df = pd.DataFrame({'Symbol': edge_names, 'U_risk': final_actions})
    action_df['Abs_U'] = action_df['U_risk'].abs()
    action_df = action_df.sort_values(by='Abs_U', ascending=False)
    
    execute_action_tensor(action_df, active_edges, current_phi)
    
    pe.mt5.shutdown()

def simulate_virtual_trade(action_df, previous_portfolio, current_prices, virtual_equity, current_phi, S_norm, dJ):
    """
    Virtual Execution Ledger for backtesting.
    Returns:
      pnl: float
      new_portfolio: dict mapping symbol to lots (signed for long/short)
      logs: list of dicts for the CSV
    """
    if current_phi < 0.5:
        # Volcanic Eruption: liquidate all positions and hold cash
        action_df = pd.DataFrame({'Symbol': [], 'U_risk': [], 'Abs_U': []})
        
    valid_signals = action_df[action_df['Abs_U'] >= CONVICTION_THRESHOLD].head(MAX_POSITIONS)
    
    total_risk_dollars = virtual_equity * TARGET_RISK_PCT
    total_conviction = valid_signals['Abs_U'].sum() if not valid_signals.empty else 1.0

    target_portfolio = {}
    
    if not valid_signals.empty:
        for _, row in valid_signals.iterrows():
            sym = row['Symbol']
            u_risk = row['U_risk']
            
            # Simple scaling for virtual lots
            weight = row['Abs_U'] / total_conviction
            base_trade_risk = total_risk_dollars * weight
            spectral_multiplier = np.log1p(row['Abs_U']) 
            compounded_risk = base_trade_risk * spectral_multiplier
            
            # Simulated lot sizing (simplified ATR = 0.005 for backtest)
            simulated_atr = 0.005
            sl_distance = simulated_atr * 2.5
            loss_per_lot = (sl_distance / 0.00001) * 1.0 # Assuming tick size 0.00001
            
            raw_lot = compounded_risk / loss_per_lot
            lot = max(0.01, round(raw_lot, 2))
            
            direction = 1 if u_risk > 0 else -1
            target_portfolio[sym] = lot * direction

    # Calculate Mark-to-Market PnL on previous positions transitioning to target_portfolio
    step_pnl = 0.0
    logs = []
    
    # We evaluate what happened to the positions we held from the LAST step
    # Wait, the prices we have are current prices. To calculate PnL, we need previous prices!
    # For a simplified backtest simulation:
    for sym, pos_size in previous_portfolio.items():
        if sym in current_prices and 'prev_' + sym in current_prices:
            curr_p = current_prices[sym]
            prev_p = current_prices['prev_' + sym]
            
            # Points gained/lost
            delta_price = curr_p - prev_p
            # 1 lot = 100,000 units usually. Simple PnL:
            trade_pnl = pos_size * 100000 * delta_price
            step_pnl += trade_pnl
            
            # Log for the audit script
            S_norm_1d = np.atleast_1d(S_norm)
            dJ_1d = np.atleast_1d(dJ)
            
            logs.append({
                'symbol': sym,
                'pnl': trade_pnl,
                'entropy': float(S_norm_1d[sym_idx]) if sym_idx < len(S_norm_1d) else 0.0,
                'pressure_delta': float(abs(dJ_1d[sym_idx])) if sym_idx < len(dJ_1d) else 0.0
            })

    return step_pnl, target_portfolio, logs

if __name__ == "__main__":
    main()
