import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.append("e:/spiderweb_core/AG_marketmap/backend/experimental")
# Run the audit script and get the trade_log
from forensic_audit import trade_log

df = pd.DataFrame(trade_log)

print("\n=== PnL BY SYMBOL ===")
grouped = df.groupby('symbol').agg(
    Trades=('symbol', 'count'),
    WinRate=('pnl_pct', lambda x: (x > 0).mean()),
    NetPnL=('pnl_pct', 'sum'),
    AvgPnL=('pnl_pct', 'mean'),
    WorstLoss=('pnl_pct', 'min'),
    BestWin=('pnl_pct', 'max')
).sort_values(by='NetPnL', ascending=False)

# Formatting
grouped['WinRate'] = grouped['WinRate'].apply(lambda x: f"{x:.2%}")
grouped['NetPnL'] = grouped['NetPnL'].apply(lambda x: f"{x:.2%}")
grouped['AvgPnL'] = grouped['AvgPnL'].apply(lambda x: f"{x:.2%}")
grouped['WorstLoss'] = grouped['WorstLoss'].apply(lambda x: f"{x:.2%}")
grouped['BestWin'] = grouped['BestWin'].apply(lambda x: f"{x:.2%}")

print(grouped.to_string())

# Forex vs Metals
metals = ['XAUUSD', 'XAGUSD', 'XAUEUR']
forex = [s for s in df['symbol'].unique() if s not in metals]

df_metals = df[df['symbol'].isin(metals)]
df_forex = df[df['symbol'].isin(forex)]

print("\n=== MACRO SECTOR BREAKDOWN ===")
print(f"METALS (Gold/Silver):")
print(f"  Trades: {len(df_metals)}")
print(f"  Win Rate: {(df_metals['pnl_pct'] > 0).mean():.2%}")
print(f"  Net PnL (Uncompounded): {df_metals['pnl_pct'].sum():.2%}")
print(f"  Worst Loss: {df_metals['pnl_pct'].min():.2%}")

print(f"\nFOREX (Fiat Currencies):")
print(f"  Trades: {len(df_forex)}")
print(f"  Win Rate: {(df_forex['pnl_pct'] > 0).mean():.2%}")
print(f"  Net PnL (Uncompounded): {df_forex['pnl_pct'].sum():.2%}")
print(f"  Worst Loss: {df_forex['pnl_pct'].min():.2%}")
