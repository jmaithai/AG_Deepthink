import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# AION LAG-CATCH ENGINE
# ============================================================
# SYNTHESIS: AION's manifold filter + MODERN statarb's direction
#
# The 68-D eigenvector decomposition identifies:
#   Epicenter = max eigenvector loading = THE LEADER
#   Outlet    = min eigenvector loading = THE LAGGARD
#
# When S < 3.5 AND Z < 0.3:
#   Manifold collapsed + outlet maximally decoupled
#   = Structural lag at maximum stretch
#
# The trade: Outlet FOLLOWS the epicenter (not against it)
#   Epicenter velocity UP   -> BUY  outlet (it must catch up)
#   Epicenter velocity DOWN -> SELL outlet (it must catch up)
# ============================================================

def run_lag_catch_engine():
    file_path = "archive/fractal_manifold_6M.parquet"
    print(f"[*] AION LAG-CATCH ENGINE: Loading 6-Month Manifold...")

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Error: {e}")
        return

    vol_cols   = [c for c in df.columns if c.endswith('_M1_vol')]
    close_cols = [c for c in df.columns if c.endswith('_M1_close')]
    symbols    = [c.replace('_M1_close', '') for c in close_cols]

    # --- EVENT CLOCK (25,000 volume units) ---
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

    # --- KINEMATICS ---
    log_prices = np.log(sensorium[close_cols].values.astype(float) + 1e-9)
    velocity   = np.diff(log_prices, axis=0, prepend=log_prices[0:1])

    WINDOW  = 21
    raw_df  = df.copy()
    events  = []

    print("[*] Scanning for Manifold Collapse Events (S < 3.5, Z < 0.3)...")

    for i in range(WINDOW, n_states):
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

        if S < 3.5 and Z < 0.3:
            # ---- THE SYNTHESIS ----
            # Epicenter velocity = direction the LEADER is moving RIGHT NOW
            epi_velocity = velocity[i, epi_idx]

            # THE CORRECT TRADE: outlet must FOLLOW the epicenter (lag-catch-up)
            # NOT reverse, NOT undefined — same direction as the leader
            trade_direction = np.sign(epi_velocity) if epi_velocity != 0 else 1.0

            # Secondary metric: how stretched is the lag?
            # Higher |epi_velocity| relative to |out_velocity| = deeper lag = stronger signal
            out_velocity = velocity[i, out_idx]
            lag_magnitude = abs(epi_velocity) / (abs(out_velocity) + 1e-10)

            events.append({
                'State_ID'       : i,
                'Time'           : sensorium['chronological_time'].iloc[i],
                'Entropy'        : S,
                'Z'              : Z,
                'Epicenter'      : symbols[epi_idx],
                'Outlet'         : symbols[out_idx],
                'Epi_Velocity'   : epi_velocity,
                'Out_Velocity'   : out_velocity,
                'Lag_Magnitude'  : lag_magnitude,
                'Trade_Direction': trade_direction,
            })

    if not events:
        print("[-] No manifold collapse events found.")
        return

    events_df = pd.DataFrame(events)
    events_df['HourGroup'] = events_df['Time'].dt.floor('H')

    # Deduplicate per pair per hour (same as resonance_audit baseline)
    distinct = events_df.drop_duplicates(subset=['HourGroup', 'Outlet'])
    total    = len(distinct)

    print(f"\n[+] Discovered {total} distinct Lag-Catch events over 6 months.")

    # Show top 5 deepest lag events
    top5 = distinct.nlargest(5, 'Lag_Magnitude')
    print(f"\n[*] Top 5 deepest lag events (epicenter running hardest vs frozen outlet):")
    for _, row in top5.iterrows():
        arrow = '↑' if row['Trade_Direction'] > 0 else '↓'
        print(f"    {row['Time'].strftime('%Y-%m-%d %H:%M')} | {row['Epicenter']}->{row['Outlet']} "
              f"| EpiV={row['Epi_Velocity']:+.5f} OutV={row['Out_Velocity']:+.6f} "
              f"| Lag={row['Lag_Magnitude']:.1f}x | Trade: {arrow}")

    print(f"\n[*] Direction breakdown:")
    longs  = (distinct['Trade_Direction'] > 0).sum()
    shorts = (distinct['Trade_Direction'] < 0).sum()
    print(f"    LONG (outlet follows epicenter UP):   {longs}")
    print(f"    SHORT (outlet follows epicenter DOWN): {shorts}")

    print(f"\n[*] Initiating Thermodynamic Forward-Walk Falsification...")
    print(f"    Walk = 100,000 volume units | Win = release > 2x friction\n")

    passes, failures = 0, 0
    release_log, loss_log = [], []
    pair_results = {}

    for _, event in distinct.iterrows():
        t0        = event['Time']
        outlet    = event['Outlet']
        direction = event['Trade_Direction']
        epi       = event['Epicenter']

        post = raw_df[raw_df.index >= t0].copy()
        if len(post) < 5:
            continue

        post['post_energy'] = post[vol_cols].sum(axis=1).cumsum()
        forward_df = post[post['post_energy'] <= 100000]
        if len(forward_df) < 5:
            continue

        p0   = forward_df[f"{outlet}_M1_close"].iloc[0]
        disp = np.log(forward_df[f"{outlet}_M1_close"] / p0) * 100

        if direction > 0:   # long the outlet (catching up to epicenter running up)
            release  = disp.max()
            friction = abs(disp.min())
        else:               # short the outlet (catching up to epicenter running down)
            release  = abs(disp.min())
            friction = max(disp.max(), 0.0)
        friction = abs(friction)

        pair_key = f"{epi}→{outlet}"
        if pair_key not in pair_results:
            pair_results[pair_key] = {'p': 0, 'f': 0}

        if release > 0 and release > (friction * 2):
            passes += 1
            release_log.append(release)
            pair_results[pair_key]['p'] += 1
        else:
            failures += 1
            loss_log.append(friction)
            pair_results[pair_key]['f'] += 1

    if passes + failures == 0:
        print("[-] No events had sufficient forward data.")
        return

    win_rate  = passes / (passes + failures) * 100
    avg_rel   = np.mean(release_log) if release_log else 0.0
    avg_loss  = np.mean(loss_log)    if loss_log    else 0.0
    exp_ratio = avg_rel / (avg_loss + 0.0001) if passes > 0 else 0.0
    ev        = (win_rate/100 * avg_rel) - ((1 - win_rate/100) * avg_loss)

    # Per-pair forensics
    print("[*] Per-Pair Results:")
    for pair, res in sorted(pair_results.items(), key=lambda x: -(x[1]['p'] + x[1]['f'])):
        total_pair = res['p'] + res['f']
        wr = res['p'] / total_pair * 100 if total_pair > 0 else 0
        print(f"    {pair:<25} | {total_pair:>3} events | Win: {wr:.1f}%")

    print("\n" + "="*80)
    print("AION LAG-CATCH ENGINE: 6-MONTH DIRECTION-AWARE AUDIT")
    print("Doctrine: Outlet follows epicenter | Entry: same dir as epicenter velocity")
    print("="*80)
    print(f"Total Lag-Catch Events:            {passes + failures}")
    print(f"Falsification Passes:              {passes}")
    print(f"Falsification Failures:            {failures}")
    print(f"\n[ LAG-CATCH DOCTRINE METRICS ]")
    print(f"Mathematical Win Rate:     {win_rate:.2f}%")
    print(f"Average Kinetic Release:   {avg_rel:.4f}%  (Yield on Success)")
    print(f"Average Adverse Friction:  {avg_loss:.4f}%  (Drawdown on Failure)")
    print(f"System Expectancy Ratio:   {exp_ratio:.2f}x")
    print(f"Expected Value (EV):       {ev:+.5f}% per event")
    print("="*80)

    print(f"\n[ COMPLETE DOCTRINE LEADERBOARD ]")
    print(f"  Entropy-Drop            → Win: 16.18% | EV: +0.02925% | Exp: 23.40x")
    print(f"  Z Locked Capacitor      → Win: 18.75% | EV: +0.10770% | Exp: 43.32x")
    print(f"  Epicenter Reversal      → Win: 40.62% | EV: -0.00886% | Exp:  1.19x")
    print(f"  Hamiltonian Coil        → Win: 41.06% | EV: -0.00787% | Exp:  1.16x")
    print(f"  *** LAG-CATCH (THIS)    → Win: {win_rate:.2f}%  | EV: {ev:+.5f}% | Exp: {exp_ratio:.2f}x")

    print(f"\n[ VERDICT ]")
    if win_rate > 50 and ev > 0.05:
        print("  *** EUREKA CONFIRMED. LAG-CATCH DOMINATES. THE SYNTHESIS WORKS. ***")
        print("  Outlet follows the epicenter. The lag is the signal. Direction is the key.")
    elif ev > 0.10 and exp_ratio > 20:
        print("  POSITIVE CONVEXITY AMPLIFIED. LAG-CATCH SUPERIOR TO ALL PRIOR DOCTRINES.")
    elif ev > 0 and ev >= 0.10:
        print("  POSITIVE EV. Comparable to Z-Alone. Calibrate for improvement.")
    elif ev > 0:
        print("  WEAK POSITIVE EV. Directional approach valid but needs further filtering.")
    else:
        print("  NEGATIVE EV. Direction alone insufficient — lag detection needs refinement.")

if __name__ == "__main__":
    run_lag_catch_engine()
