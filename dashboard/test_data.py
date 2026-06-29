import MetaTrader5 as mt5
import sys

path = sys.argv[1]
print(f"Connecting to {path}...")
mt5.initialize(path=path)
syms = mt5.symbols_get()
valid = [s.name for s in syms if s.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL] if syms else []
fringe = valid[:5]
print("Symbols:", fringe)

prices = []
for s in fringe:
    t = mt5.symbol_info_tick(s)
    prices.append(t.bid if t else 1.0)
print("Prices:", prices)

import numpy as np
raw = np.array([prices, prices])
mean = np.mean(raw, axis=0)
std = np.std(raw, axis=0) + 1e-9
norm = (raw - mean) / std
vel = np.diff(norm, axis=0, prepend=norm[0:1])[-1]
disp = norm[-1] - np.mean(norm[-1])
V = 0.5 * disp**2
T = 0.5 * vel**2

print("V:", V)
print("T:", T)

mt5.shutdown()
