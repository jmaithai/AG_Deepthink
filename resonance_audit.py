import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def run_resonance_audit():
    file_path = "archive/fractal_manifold_6M.parquet"
    print(f"[*] RESONANCE AUDIT: Loading 6-Month Manifold from {file_path}")

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Error: {e}")
        return

    print("[*] Rebuilding Event Clock (25,000 volume units)...")
    vol_cols  = [c for c in df.columns if c.endswith('_M1_vol')]
    close_cols = [c for c in df.columns if c.endswith('_M1_close')]
    symbols = [c.replace('_M1_close', '') for c in close_cols]

    df['network_ticks'] = df[vol_cols].sum(axis=1)
    df['cumulative_ticks'] = df['network_ticks'].cumsum()
    df['Event_Clock'] = (df['cumulative_ticks'] // 25000).astype(int)
    df['chronological_time'] = df.index

    sensorium = df.groupby('Event_Clock').last()
    print(f"[+] Compressed {len(df):,} bars into {len(sensorium):,} Event States.")

    print("[*] Calculating Kinematics...")
    prices   = np.log(sensorium[close_cols].values + 1e-9)
    velocity = np.diff(prices, axis=0, prepend=prices[0:1])

    print("[*] Scanning for Locked Capacitor States (Z < 0.3 AND S < 3.5)...")
    window = 21
    capacitor_states = []
    raw_df = df.copy()

    for i in range(window, len(sensorium)):
        v_window = velocity[i-window:i]

        # Z-Score normalisation
        mean_v = np.mean(v_window, axis=0)
        std_v  = np.std(v_window, axis=0) + 1e-12
        v_norm = (v_window - mean_v) / std_v

        cov_matrix = np.cov(v_norm.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Reality Degeneracy (Shannon Entropy)
        total    = np.sum(np.abs(eigenvalues)) + 1e-12
        p_energy = np.clip(np.abs(eigenvalues) / total, 1e-12, 1.0)
        entropy  = -np.sum(p_energy * np.log2(p_energy))

        # Impedance Ratio Z
        dom_vec = eigenvectors[:, 0]
        epi_idx = int(np.argmax(np.abs(dom_vec)))
        out_idx = int(np.argmin(np.abs(dom_vec)))

        epi_vol = np.std(v_norm[:, epi_idx])
        out_vol = np.std(v_norm[:, out_idx])
        Z = out_vol / (epi_vol + 1e-9)

        # LOCKED CAPACITOR CONDITION
        if entropy < 3.5 and Z < 0.3:
            capacitor_states.append({
                'State_ID'  : i,
                'Time'      : sensorium['chronological_time'].iloc[i],
                'Entropy'   : entropy,
                'Z'         : Z,
                'Epicenter' : symbols[epi_idx],
                'Outlet'    : symbols[out_idx],
            })

    if not capacitor_states:
        print("[-] No Locked Capacitor states found. Try relaxing thresholds.")
        return

    events_df = pd.DataFrame(capacitor_states)
    events_df['HourGroup'] = events_df['Time'].dt.floor('H')
    distinct = events_df.drop_duplicates(subset=['HourGroup', 'Epicenter'])

    total = len(distinct)
    print(f"\n[+] Discovered {total} distinct Locked Capacitor events over 6 months.")
    print("[*] Initiating Thermodynamic Forward-Walk Falsification...")

    passes, failures = 0, 0
    release_log, loss_log = [], []

    for _, event in distinct.iterrows():
        t0     = event['Time']
        epi    = event['Epicenter']
        outlet = event['Outlet']

        # Energy-based forward walk: 100,000 volume units post-event
        post = raw_df[raw_df.index >= t0].copy()
        if len(post) < 5:
            continue

        post['post_energy'] = post[vol_cols].sum(axis=1).cumsum()
        forward_df = post[post['post_energy'] <= 100000]
        if len(forward_df) < 5:
            continue

        p0_epi = forward_df[f"{epi}_M1_close"].iloc[0]
        p0_out = forward_df[f"{outlet}_M1_close"].iloc[0]

        epi_disp = np.log(forward_df[f"{epi}_M1_close"] / p0_epi) * 100
        out_disp = np.log(forward_df[f"{outlet}_M1_close"] / p0_out) * 100

        max_epi = epi_disp.max() if abs(epi_disp.max()) > abs(epi_disp.min()) else epi_disp.min()

        max_out_up   = out_disp.max()
        max_out_down = out_disp.min()

        if abs(max_out_down) > abs(max_out_up):
            release = abs(max_out_down)
            friction = abs(max_out_up)
        else:
            release = abs(max_out_up)
            friction = abs(max_out_down)

        # AION Doctrine falsification: Release > Illusion AND Release > 2x Friction
        if release > abs(max_epi) and release > (friction * 2):
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
    print("RESONANCE AUDIT: 6-MONTH LOCKED CAPACITOR DOCTRINE (Z < 0.3 | S < 3.5)")
    print("="*80)
    print(f"Total Locked Capacitor Events:     {passes + failures}")
    print(f"Falsification Passes:              {passes}")
    print(f"Falsification Failures:            {failures}")
    print(f"\n[ DOCTRINE METRICS ]")
    print(f"Mathematical Win Rate:     {win_rate:.2f}%")
    print(f"Average Kinetic Release:   {avg_rel:.4f}%  (Yield on Success)")
    print(f"Average Adverse Friction:  {avg_loss:.4f}%  (Drawdown on Failure)")
    print(f"System Expectancy Ratio:   {exp_ratio:.2f}x")
    print(f"Expected Value (EV):       {ev:+.5f}% per event")
    print("="*80)

    if ev > 0 and exp_ratio > 2.0:
        print("[ VERDICT ] POSITIVE CONVEXITY. THE LOCKED CAPACITOR LAW IS PROVEN.")
    elif ev > 0:
        print("[ VERDICT ] WEAK POSITIVE EV. Signal exists — tighten thresholds.")
    else:
        print("[ VERDICT ] NEGATIVE EV. DOCTRINE REQUIRES CALIBRATION.")

if __name__ == "__main__":
    run_resonance_audit()
