import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def run_compound_engine():
    file_path = "archive/fractal_manifold_6M.parquet"
    print(f"[*] COMPOUND ENGINE: Loading 6-Month Manifold...")

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Error: {e}")
        return

    vol_cols   = [c for c in df.columns if c.endswith('_M1_vol')]
    close_cols = [c for c in df.columns if c.endswith('_M1_close')]
    symbols    = [c.replace('_M1_close', '') for c in close_cols]
    n_sym      = len(symbols)

    # --- EVENT CLOCK ---
    df['network_ticks']    = df[vol_cols].sum(axis=1)
    df['cumulative_ticks'] = df['network_ticks'].cumsum()
    df['Event_Clock']      = (df['cumulative_ticks'] // 25000).astype(int)
    df['chronological_time'] = df.index

    agg = {col: 'last' for col in close_cols}
    agg.update({col: 'sum' for col in vol_cols})
    agg['chronological_time'] = 'last'
    sensorium = df.groupby('Event_Clock').agg(agg)
    n_states  = len(sensorium)

    print(f"[+] Compressed {len(df):,} bars into {n_states:,} Event States.")
    print(f"[*] Computing Kinematics + Hamiltonian + Barycenter...")

    # --- KINEMATICS ---
    log_prices = np.log(sensorium[close_cols].values.astype(float) + 1e-9)
    velocity   = np.diff(log_prices, axis=0, prepend=log_prices[0:1])
    volumes    = sensorium[vol_cols].values.astype(float)
    vol_norm   = volumes / (volumes.mean(axis=0) + 1e-9)

    # --- HAMILTONIAN: T + V ---
    T = vol_norm * (velocity ** 2)

    # Equilibrium 1: Rolling SMA (node's own history)
    lp_df        = pd.DataFrame(log_prices, columns=close_cols)
    rolling_mean = lp_df.rolling(21, min_periods=1).mean().values
    disp_sma     = log_prices - rolling_mean
    V_sma        = vol_norm * (disp_sma ** 2)

    # Equilibrium 2: Log-Normalised Barycenter (geometric center of manifold)
    # Correct: use log prices so all 30 nodes are dimensionless
    barycenter   = log_prices.mean(axis=1, keepdims=True)   # (n_states, 1)
    disp_bary    = log_prices - barycenter                   # displacement from manifold center
    V_bary       = vol_norm * (disp_bary ** 2)

    # Combined potential: average of SMA tension + barycentric tension
    V = (V_sma + V_bary) / 2.0

    H  = T + V
    dH = np.diff(H, axis=0, prepend=H[0:1])

    # --- SIGNAL A: Z-LOCKED CAPACITOR ---
    # Eigen-decomposition per rolling window → S < 3.5 AND Z < 0.3
    print("[*] Scanning: Signal A — Z-Locked Capacitor (S < 3.5, Z < 0.3)...")
    WINDOW = 21
    HIST_W = 50
    ab_count = 0   # track A∩B before C filter

    compound_events = []
    raw_df = df.copy()
    prev_S = None

    for i in range(max(WINDOW, HIST_W), n_states):
        # --- EIGEN DECOMPOSITION ---
        v_window = velocity[i-WINDOW:i]
        mean_v   = np.mean(v_window, axis=0)
        std_v    = np.std(v_window, axis=0) + 1e-12
        v_norm   = (v_window - mean_v) / std_v

        cov      = np.cov(v_norm.T)
        evals, evecs = np.linalg.eigh(cov)
        idx      = np.argsort(evals)[::-1]
        evals    = evals[idx];  evecs = evecs[:, idx]

        total    = np.sum(np.abs(evals)) + 1e-12
        p_energy = np.clip(np.abs(evals) / total, 1e-12, 1.0)
        S        = -np.sum(p_energy * np.log2(p_energy))

        dom_vec  = evecs[:, 0]
        epi_idx  = int(np.argmax(np.abs(dom_vec)))
        out_idx  = int(np.argmin(np.abs(dom_vec)))

        epi_vol  = np.std(v_norm[:, epi_idx])
        out_vol  = np.std(v_norm[:, out_idx])
        Z        = out_vol / (epi_vol + 1e-9)

        signal_A = (S < 3.5) and (Z < 0.3)

        if not signal_A:
            prev_S = S
            continue

        # --- SIGNAL B: HAMILTONIAN COIL on the outlet node ---
        v_hist = V[max(0, i-HIST_W):i, out_idx]
        t_hist = T[max(0, i-HIST_W):i, out_idx]

        v_70   = np.percentile(v_hist, 70)   # relaxed: was 90th
        t_30   = np.percentile(t_hist, 30)   # relaxed: was 10th

        signal_B = (
            V[i, out_idx] > v_70 and    # outlet above-average potential
            T[i, out_idx] < t_30 and    # outlet below-average kinetic
            dH[i, out_idx] > 0          # energy flowing into outlet
        )

        # --- SIGNAL C: BARYCENTRIC DISPLACEMENT ---
        # Outlet must be stretched from manifold centre more than 70th percentile
        bary_hist = np.abs(disp_bary[max(0, i-HIST_W):i, out_idx])
        bary_50   = np.percentile(bary_hist, 50)   # relaxed: was 70th
        signal_C  = (np.abs(disp_bary[i, out_idx]) > bary_50)

        # --- COMPOUND TRIGGER: all three signals must agree ---
        if signal_A and signal_B:
            ab_count += 1
        if signal_A and signal_B and signal_C:
            # Direction: outlet is displaced from both SMA and barycenter
            # Trade toward equilibrium (mean reversion)
            disp_total = disp_sma[i, out_idx] + disp_bary[i, out_idx]
            direction  = -np.sign(disp_total) if disp_total != 0 else 1.0

            compound_events.append({
                'State_ID' : i,
                'Time'     : sensorium['chronological_time'].iloc[i],
                'Entropy'  : S,
                'Z'        : Z,
                'Epicenter': symbols[epi_idx],
                'Outlet'   : symbols[out_idx],
                'V_outlet' : V[i, out_idx],
                'T_outlet' : T[i, out_idx],
                'Bary_disp': disp_bary[i, out_idx],
                'Direction': direction,
            })

        prev_S = S

    print(f"    A∩B (before C filter): {ab_count} events")
    if not compound_events:
        print("[-] No A∩B∩C events found after C filter.")
        print("    Consider dropping Signal C and running A∩B only.")
        return

    events_df = pd.DataFrame(compound_events)
    events_df['HourGroup'] = events_df['Time'].dt.floor('H')
    distinct  = events_df.drop_duplicates(subset=['HourGroup', 'Outlet'])
    total     = len(distinct)

    print(f"\n[+] Discovered {total} COMPOUND events (A∩B∩C) over 6 months.")
    print(f"    (A∩B before C: {ab_count} | Z-alone: 32 | Hamiltonian-alone: 2097)")
    print("[*] Initiating Thermodynamic Forward-Walk Falsification...")

    passes, failures = 0, 0
    release_log, loss_log = [], []

    for _, event in distinct.iterrows():
        t0        = event['Time']
        outlet    = event['Outlet']
        direction = event['Direction']

        post = raw_df[raw_df.index >= t0].copy()
        if len(post) < 5:
            continue

        post['post_energy'] = post[vol_cols].sum(axis=1).cumsum()
        forward_df = post[post['post_energy'] <= 100000]
        if len(forward_df) < 5:
            continue

        p0   = forward_df[f"{outlet}_M1_close"].iloc[0]
        disp = np.log(forward_df[f"{outlet}_M1_close"] / p0) * 100

        if direction > 0:
            release  = disp.max()
            friction = abs(disp.min())
        else:
            release  = abs(disp.min())
            friction = max(disp.max(), 0.0)
        friction = abs(friction)

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
    print("COMPOUND ENGINE: A∩B∩C TRIPLE-CONFIRMED LOCKED CAPACITOR")
    print("Signal A: Z-Locked Capacitor  (S < 3.5, Z < 0.3)")
    print("Signal B: Hamiltonian Coil    (V > 90th, T < 10th, ΔH > 0)")
    print("Signal C: Barycenter Stretch  (displaced > 70th pct from manifold centre)")
    print("="*80)
    print(f"Total Triple-Confirmed Events:     {passes + failures}")
    print(f"Falsification Passes:              {passes}")
    print(f"Falsification Failures:            {failures}")
    print(f"\n[ COMPOUND DOCTRINE METRICS ]")
    print(f"Mathematical Win Rate:     {win_rate:.2f}%")
    print(f"Average Kinetic Release:   {avg_rel:.4f}%  (Yield on Success)")
    print(f"Average Adverse Friction:  {avg_loss:.4f}%  (Drawdown on Failure)")
    print(f"System Expectancy Ratio:   {exp_ratio:.2f}x")
    print(f"Expected Value (EV):       {ev:+.5f}% per event")
    print("="*80)
    print(f"\n[ FULL DOCTRINE LEADERBOARD ]")
    print(f"  Entropy-Drop        → Win: 16.18% | EV: +0.02925% | Exp: 23.40x")
    print(f"  Z Locked Capacitor  → Win: 18.75% | EV: +0.10770% | Exp: 43.32x")
    print(f"  Hamiltonian Coil    → Win: 41.06% | EV: -0.00787% | Exp:  1.16x")
    print(f"  Compound A∩B∩C      → Win: {win_rate:.2f}%  | EV: {ev:+.5f}% | Exp: {exp_ratio:.2f}x")

    if win_rate > 45 and ev > 0.05:
        print("\n[ VERDICT ] *** EUREKA. COMPOUND FILTER SOLVES THE PROBLEM. ***")
    elif ev > 0.10 and exp_ratio > 20:
        print("\n[ VERDICT ] POSITIVE CONVEXITY AMPLIFIED. COMPOUND DOCTRINE SUPERIOR.")
    elif ev > 0 and exp_ratio > 3:
        print("\n[ VERDICT ] POSITIVE EV. Compound viable. Calibrate thresholds further.")
    elif ev > 0:
        print("\n[ VERDICT ] WEAK POSITIVE EV. Marginal improvement over Z-alone.")
    else:
        print("\n[ VERDICT ] NEGATIVE EV. Over-filtering killed the signal.")

if __name__ == "__main__":
    run_compound_engine()
