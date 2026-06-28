import pandas as pd
import json
import os

def audit_backtest(log_file):
    if not os.path.exists(log_file):
        print(f"[!] Log file '{log_file}' not found. We need to run a real backtest first to generate it.")
        return

    # This assumes your backtest logs are in JSON/CSV format
    df = pd.read_csv(log_file)
    
    # 1. Isolate the losers
    losses = df[df['pnl'] < 0]
    
    # 2. Check for Entropy Trap
    entropy_traps = losses[losses['entropy'] > 0.85]
    
    # 3. Check for Topology Lag
    # Did the Pressure (P) change significantly between Signal and Exit?
    lag_losses = losses[losses['pressure_delta'] > 2.0]
    
    print(f"Total Trades: {len(df)}")
    print(f"Entropy Traps: {len(entropy_traps)} (Reduce Threshold to avoid)")
    print(f"Topology Lag: {len(lag_losses)} (Increase polling frequency)")
    print(f"Friction Losses: {len(losses) - len(entropy_traps) - len(lag_losses)}")

if __name__ == "__main__":
    # Run this to see the truth
    audit_backtest('backtest.log')
