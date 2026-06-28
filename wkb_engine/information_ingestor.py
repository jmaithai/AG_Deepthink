import os
import sys
import time
import re
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional
import MetaTrader5 as mt5

class InformationIngestor:
    def __init__(self, mt5_path: Optional[str] = None):
        self.mt5_path = mt5_path
        self.connected = False
        
    def connect(self) -> bool:
        if self.connected:
            return True
        if self.mt5_path:
            if not mt5.initialize(path=self.mt5_path):
                print(f"Failed to initialize MT5 at path {self.mt5_path}: {mt5.last_error()}", file=sys.stderr)
                return False
        else:
            if not mt5.initialize():
                print(f"Failed to initialize MT5: {mt5.last_error()}", file=sys.stderr)
                return False
        self.connected = True
        return True
        
    def shutdown(self):
        if self.connected:
            mt5.shutdown()
            self.connected = False
            
    def discover_active_symbols(self) -> List[str]:
        """Discovers all active tradeable symbols from MT5, skipping stocks/shares."""
        if not self.connect():
            return []
        
        all_syms = mt5.symbols_get()
        if all_syms is None:
            return []
            
        active = []
        for s in all_syms:
            if s.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
                continue
            path = s.path.lower()
            if any(w in path for w in ["shares", "stock", "equity", "equities", "cfd-single-stocks"]):
                continue
            active.append(s.name)
        return active

    def parse_symbol_nodes(self, name: str) -> Tuple[str, Optional[str]]:
        """Parses raw symbol name to return the underlying base and quote nodes."""
        name_clean = re.sub(r'[^A-Z0-9]', '', name.upper())
        
        # Metals
        if any(m in name_clean for m in ["XAU", "GOLD"]):
            return "GOLD", self._extract_quote(name_clean, "GOLD")
        if any(m in name_clean for m in ["XAG", "SILVER"]):
            return "SILVER", self._extract_quote(name_clean, "SILVER")
        if "XPT" in name_clean:
            return "PLATINUM", self._extract_quote(name_clean, "XPT")
        if "XPD" in name_clean:
            return "PALLADIUM", self._extract_quote(name_clean, "XPD")
            
        # Energies & Indices
        for idx in ["US500", "SPX", "NAS100", "USTEC", "US30", "DJI", "JPN225", "UK100", "AUS200", "HK50", "FRA40", "GER40", "EUSTX50"]:
            if idx in name_clean:
                return idx, "USD"
        if "BRENT" in name_clean:
            return "BRENT", "USD"
        if "WTI" in name_clean or "CRUDE" in name_clean:
            return "CRUDE", "USD"
        if "NATGAS" in name_clean or "GAS" in name_clean:
            return "NATGAS", "USD"
            
        # Forex / Cryptocurrencies
        if len(name_clean) >= 6:
            return name_clean[:3], name_clean[3:6]
        return name_clean, None

    def _extract_quote(self, name: str, base: str) -> str:
        for fiat in ["USD", "EUR", "GBP", "AUD", "NZD", "JPY", "CHF", "CAD", "HKD"]:
            if name.endswith(fiat):
                return fiat
        return "USD"

    def compute_proper_time_steps(self, prices: np.ndarray, window: int = 24) -> np.ndarray:
        """
        Computes dynamic proper-time steps dtau based on rolling Shannon entropy of returns.
        dtau_t = (Entropy_t / mean(Entropy)) * dt
        """
        if len(prices) < window + 2:
            return np.ones(len(prices))
            
        # Compute log returns
        returns = np.diff(np.log(prices))
        
        entropies = []
        for i in range(len(returns)):
            if i < window:
                entropies.append(1.0)
                continue
            sub = returns[i - window : i]
            # Discretize returns into 10 bins to compute histogram entropy
            hist, _ = np.histogram(sub, bins=10)
            probs = hist / np.sum(hist)
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs))
            entropies.append(entropy)
            
        entropies = np.array(entropies)
        # Pad first element to match input length
        entropies = np.insert(entropies, 0, entropies[0])
        
        # Calculate rolling mean of entropy to normalize
        rolling_mean = pd.Series(entropies).rolling(window * 4, min_periods=window).mean().fillna(1.0).values
        
        # dtau is the relative density of information arrival
        dtau = entropies / np.maximum(rolling_mean, 1e-6)
        return np.clip(dtau, 0.05, 5.0)

    def fetch_tick_history(self, symbol: str, start_time: datetime, count: int = 100000) -> Optional[pd.DataFrame]:
        """Fetches raw tick events from MT5 starting from a specific time."""
        if not self.connect():
            return None
            
        ticks = mt5.copy_ticks_from(symbol, start_time, count, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            return None
            
        df = pd.DataFrame(ticks)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['price'] = 0.5 * (df['bid'] + df['ask'])
        return df[['time', 'price', 'volume']]
