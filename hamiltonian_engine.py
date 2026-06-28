import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def run_hamiltonian_engine():
    file_path = "archive/fractal_manifold_6M.parquet"
    print(f"[*] HAMILTONIAN ENGINE: Loading 6-Month Manifold...")

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Error: {e}")
        return

    vol_cols   = [c for c in df.columns if c.endswith('_M1_vol')]
    close_cols = [c for c in df.columns if c.endswith('_M1_close')]
    symbols    = [c.replace('_M1_close', '') for c in close_cols]

    # --- EVENT CLOCK ---
    df['network_ticks']    = df[vol_cols].sum(axis=1)
    df['cumulative_ticks'] = df['network_ticks'].cumsum()
    df['Event_Clock']      = (df['cumulative_ticks'] // 25000).astype(int)
    df['chronological_time'] = df.index

    # Aggregate sensorium: last close, sum volume per event state
    agg_dict = {col: 'last' for col in close_cols}
    agg_dict.update({col: 'sum'  for col in vol_cols})
    agg_dict['chronological_time'] = 'last'
    sensorium = df.groupby('Event_Clock').agg(agg_dict)

    n_states  = len(sensorium)
    n_symbols = len(symbols)
    print(f"[+] Compressed {len(df):,} bars into {n_states:,} Event States.")

    # --- KINEMATICS ---
    log_prices = np.log(sensorium[close_cols].values.astype(float) + 1e-9)
    velocity   = np.diff(log_prices, axis=0, prepend=log_prices[0:1])
    volumes    = sensorium[vol_cols].values.astype(float)

    # Normalise volumes per symbol so FX/Indices are comparable
    vol_mean = volumes.mean(axis=0) + 1e-9
    vol_norm = volumes / vol_mean

    # ================================================================
    # HAMILTONIAN PHYSICS
    # H = T + V
    # T (Kinetic)   = vol × velocity²      — money already moving
    # V (Potential) = vol × displacement²  — money compressed, not yet released
    # ================================================================
    T = vol_norm * (velocity ** 2)

    # Rolling equilibrium = 21-state SMA of log prices
    SMA_WINDOW = 21
    lp_df        = pd.DataFrame(log_prices, columns=close_cols)
    rolling_mean = lp_df.rolling(SMA_WINDOW, min_periods=1).mean().values
    displacement = log_prices - rolling_mean                   # signed distance from equilibrium
    V  = vol_norm * (displacement ** 2)

    H  = T + V                                                 # total energy per node
    dH = np.diff(H, axis=0, prepend=H[0:1])                  # energy flow (+ = absorbing)

    VT_ratio = V / (T + 1e-15)                                # potential dominance

    print("[*] Scanning for Coiled Spring States...")
    print("    Signal: V > 90th pct | T < 10th pct | ΔH > 0 (absorbing energy)")

    HISTORY_WINDOW = 50
    coil_events = []
    raw_df = df.copy()

    for i in range(HISTORY_WINDOW, n_states):
        v_hist = V[max(0, i - HISTORY_WINDOW):i]
        t_hist = T[max(0, i - HISTORY_WINDOW):i]

        v_90 = np.percentile(v_hist, 90, axis=0)   # high potential threshold
        t_10 = np.percentile(t_hist, 10, axis=0)   # low kinetic threshold

        is_coiled = (
            (V[i] > v_90) &      # unusually compressed (high potential)
            (T[i] < t_10) &      # barely moving (low kinetic)
            (dH[i] > 0)          # absorbing energy from manifold
        )

        if not is_coiled.any():
            continue

        # Score: highest VT_ratio among coiled nodes
        coil_scores = VT_ratio[i] * is_coiled.astype(float)
        best_node   = int(np.argmax(coil_scores))

        if coil_scores[best_node] == 0:
            continue

        # Direction: node is compressed away from equilibrium → mean reversion
        disp_sign       = np.sign(displacement[i, best_node])
        trade_direction = -disp_sign if disp_sign != 0 else 1.0  # bet toward equilibrium

        coil_events.append({
            'State_ID'       : i,
            'Time'           : sensorium['chronological_time'].iloc[i],
            'Node'           : symbols[best_node],
            'VT_Ratio'       : VT_ratio[i, best_node],
            'V'              : V[i, best_node],
            'T'              : T[i, best_node],
            'dH'             : dH[i, best_node],
            'Displacement'   : displacement[i, best_node],
            'Trade_Direction': trade_direction,
        })

    if not coil_events:
        print("[-] No Coiled Spring states found. Relax thresholds.")
        return

    events_df = pd.DataFrame(coil_events)
    events_df['HourGroup'] = events_df['Time'].dt.floor('H')
    distinct  = events_df.drop_duplicates(subset=['HourGroup', 'Node'])
    total     = len(distinct)

    print(f"\n[+] Discovered {total} distinct Coiled Spring events over 6 months.")
    print(f"[*] Top 5 most coiled nodes:")
    top_nodes = distinct.nlargest(5, 'VT_Ratio')[['Node','VT_Ratio','V','T','Displacement']]
    for _, row in top_nodes.iterrows():
        print(f"    {row['Node']:<12} VT={row['VT_Ratio']:.1f}x | V={row['V']:.6f} | T={row['T']:.6f} | Disp={row['Displacement']:+.4f}")

    print("\n[*] Initiating Thermodynamic Forward-Walk Falsification...")

    passes, failures = 0, 0
    release_log, loss_log = [], []

    for _, event in distinct.iterrows():
        t0        = event['Time']
        node      = event['Node']
        direction = event['Trade_Direction']

        post = raw_df[raw_df.index >= t0].copy()
        if len(post) < 5:
            continue

        post['post_energy'] = post[vol_cols].sum(axis=1).cumsum()
        forward_df = post[post['post_energy'] <= 100000]
        if len(forward_df) < 5:
            continue

        p0   = forward_df[f"{node}_M1_close"].iloc[0]
        disp = np.log(forward_df[f"{node}_M1_close"] / p0) * 100

        if direction > 0:       # long — node was below equilibrium, expect rise
            release  = disp.max()
            friction = abs(disp.min())
        else:                   # short — node was above equilibrium, expect fall
            release  = abs(disp.min())
            friction = max(disp.max(), 0.0)

        friction = abs(friction)

        # PASS: directional release > 2x adverse friction
        if release > 0 and release > (friction * 2):
            passes += 1
            release_log.append(release)
        else:
            failures += 1
            loss_log.append(friction)

    if passes + failures == 0:
        print("[-] No events had sufficient forward data.")
        return

    win_rate  = passes / (passes + failures) * 100
    avg_rel   = np.mean(release_log) if release_log else 0.0
    avg_loss  = np.mean(loss_log)    if loss_log    else 0.0
    exp_ratio = avg_rel / (avg_loss + 0.0001) if passes > 0 else 0.0
    ev        = (win_rate/100 * avg_rel) - ((1 - win_rate/100) * avg_loss)

    print("\n" + "="*80)
    print("HAMILTONIAN ENGINE: 6-MONTH COILED SPRING AUDIT")
    print("="*80)
    print(f"Total Coiled Spring Events:        {passes + failures}")
    print(f"Falsification Passes:              {passes}")
    print(f"Falsification Failures:            {failures}")
    print(f"\n[ HAMILTONIAN DOCTRINE METRICS ]")
    print(f"Mathematical Win Rate:     {win_rate:.2f}%")
    print(f"Average Kinetic Release:   {avg_rel:.4f}%  (Yield on Success)")
    print(f"Average Adverse Friction:  {avg_loss:.4f}%  (Drawdown on Failure)")
    print(f"System Expectancy Ratio:   {exp_ratio:.2f}x")
    print(f"Expected Value (EV):       {ev:+.5f}% per event")
    print("="*80)
    print(f"\n[ FULL DOCTRINE LEADERBOARD ]")
    print(f"  Entropy-Drop            → Win: 16.18% | EV: +0.02925% | Exp: 23.40x")
    print(f"  Z Locked Capacitor      → Win: 18.75% | EV: +0.10770% | Exp: 43.32x")
    print(f"  Epicenter Reversal      → Win: 40.62% | EV: -0.00886% | Exp:  1.19x")
    print(f"  Hamiltonian Coil        → Win: {win_rate:.2f}%  | EV: {ev:+.5f}% | Exp: {exp_ratio:.2f}x")

    if win_rate > 50 and ev > 0:
        print("\n[ VERDICT ] *** EUREKA CONFIRMED. HAMILTONIAN MAP SOLVES THE PROBLEM. ***")
    elif ev > 0.05 and exp_ratio > 5.0:
        print("\n[ VERDICT ] POSITIVE CONVEXITY. HAMILTONIAN APPROACH SUPERIOR.")
    elif ev > 0:
        print("\n[ VERDICT ] POSITIVE EV. Energy mapping viable — calibrate thresholds.")
    else:
        print("\n[ VERDICT ] NEGATIVE EV. Hamiltonian alone insufficient — needs compound filter.")

if __name__ == "__main__":
    run_hamiltonian_engine()
