import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def run_truth_audit():
    file_path = "archive/omni_energy_cloud.parquet"
    print(f"[*] AION TRUTH AUDIT: Falsifying 'Outlet Locking' hypothesis...")
    
    df = pd.read_parquet(file_path)
    close_cols = [c for c in df.columns if c.endswith('_mid')]
    
    # 1. Calculate Rolling Volatility (Impedance) for all 68 nodes
    # We use a 500-tick window
    returns = np.log(df[close_cols] / df[close_cols].shift(1))
    volatility = returns.rolling(window=500).std().fillna(0)
    
    # 2. Identify "Locked" Nodes (Bottom 5% of volatility)
    threshold = volatility.quantile(0.05, axis=1)
    is_locked = (volatility.T < threshold).T
    
    # 3. Identify Systemic Kinetic Release (Top 1% of market movement)
    system_movement = volatility.mean(axis=1)
    is_rupture = system_movement > system_movement.quantile(0.99)
    
    # 4. Falsification: Does Locking precede Rupture?
    # We lag the rupture to see if Locking (T-1000) predicts Rupture (T=0)
    leads = is_locked.shift(1000)
    
    # Calculate Correlation: How often does a locked node lead a rupture?
    # We correlate the 'is_locked' boolean mask with the 'is_rupture' mask
    correlation = leads.corrwith(is_rupture, axis=0).mean()
    
    # Per-symbol breakdown: which nodes are the best predictors?
    per_sym = leads.corrwith(is_rupture, axis=0).sort_values(ascending=False)
    
    print("\n" + "="*80)
    print("AION TRUTH AUDIT: RESULTS")
    print("="*80)
    print(f"Hypothesis: 'Outlet Locking' (Near-zero vol) precedes Systemic Ruptures.")
    print(f"Systemic Manifold: 68 Assets")
    print(f"Data Points: {len(df):,} discrete events")
    print(f"\nCorrelation Coefficient: {correlation:.4f}")
    
    print(f"\n[ TOP 5 PREDICTIVE LOCK NODES ]")
    for sym, corr in per_sym.head(5).items():
        sym_name = sym.replace('_mid', '')
        print(f"  {sym_name:<12}: {corr:+.4f}")
    
    print(f"\n[ BOTTOM 5 (DEAD WEIGHT) ]")
    for sym, corr in per_sym.tail(5).items():
        sym_name = sym.replace('_mid', '')
        print(f"  {sym_name:<12}: {corr:+.4f}")
    
    if correlation > 0.4:
        print("\nRESULT: SIGNAL DETECTED. The theory is not BS.")
    elif correlation > 0.2:
        print("\nRESULT: WEAK SIGNAL. Theory has partial merit — requires refinement.")
    else:
        print("\nRESULT: NOISE DETECTED. The theory is BS. Scrapping the concept.")
    print("="*80)

if __name__ == "__main__":
    run_truth_audit()
