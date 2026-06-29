import MetaTrader5 as mt5
import time
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test_icmarkets():
    terminal_path = r"E:\ICmarkets_mt5\terminal64.exe"
    print(f"[*] Attempting to initialize ICMarkets Terminal at: {terminal_path}")
    
    if not mt5.initialize(path=terminal_path):
        print(f"[-] MT5 Init Failed. Error: {mt5.last_error()}")
        return

    print("[+] Connection Successful. Fetching Symbol Universe...")
    
    all_symbols = mt5.symbols_get()
    if all_symbols is None:
        print("[-] Failed to retrieve symbols.")
        mt5.shutdown()
        return
        
    CRYPTO_FILTER = {'UNIUSD', 'SUIUSD', 'BTCUSD', 'ETHUSD', 'SOLUSD', 'LINKUSD', 'ADAUSD', 'XAGUSD', 'XAUUSD'}
    
    print(f"[*] Total Raw Symbols found on broker: {len(all_symbols)}")
    
    candidate_symbols = []
    dropped_symbols = []
    
    for sym in all_symbols:
        # Check if visible, not crypto, and has full trade mode
        if sym.visible and sym.name not in CRYPTO_FILTER:
            if sym.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                candidate_symbols.append(sym.name)
            else:
                dropped_symbols.append((sym.name, sym.trade_mode))
                
    print("\n" + "="*80)
    print("AION MANIFOLD FILTER RESULTS (ICMARKETS)")
    print("="*80)
    print(f"Symbols accepted for full trading: {len(candidate_symbols)}")
    print(f"Symbols dropped due to trade_mode: {len(dropped_symbols)}")
    
    if len(dropped_symbols) > 0:
        print("\nSample of dropped symbols (Trade Disabled / Close Only):")
        for sym, mode in dropped_symbols[:10]:
            print(f"  - {sym} (Mode: {mode})")
            
    if len(candidate_symbols) > 0:
        print("\nSample of accepted manifold nodes:")
        print(f"  {candidate_symbols[:10]}")
        
    print("="*80)
    
    mt5.shutdown()

if __name__ == "__main__":
    test_icmarkets()
