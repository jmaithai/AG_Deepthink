import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def run_persistent_lock():
    file_path = "archive/fractal_manifold_6M.parquet"
    print(f"[*] PERSISTENT Z-LOCK ENGINE: Loading 6-Month Manifold...")

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[!] Error: {e}")
        return

    vol_cols   = [c for c in df.columns if c.endswith('_M1_vol')]
    close_cols = [c for c in df.columns if c.endswith('_M1_close')]
    symbols    = [c.replace('_M1_close', '') for c in close_cols]

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
    print(f"[*] Testing Persistent Z-Lock hypothesis...")
    print(f"    Hypothesis: Z < 0.3 held for N consecutive states = more stored pressure = higher release")

    log_prices = np.log(sensorium[close_cols].values.astype(float) + 1e-9)
    velocity   = np.diff(log_prices, axis=0, prepend=log_prices[0:1])

    WINDOW = 21
    raw_df = df.copy()

    # --- COMPUTE Z AND S FOR EVERY EVENT STATE ---
    Z_series = []
    S_series = []
    epi_series = []
    out_series = []

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

        Z_series.append({'idx': i, 'Z': Z, 'S': S,
                         'epi': symbols[epi_idx], 'out': symbols[out_idx],
                         'Time': sensorium['chronological_time'].iloc[i]})

    Z_df = pd.DataFrame(Z_series).set_index('idx')

    # --- TEST ACROSS DIFFERENT PERSISTENCE LEVELS ---
    results = {}

    for min_consecutive in [1, 2, 3, 4]:
        events = []
        lock_count = 0

        for i in range(len(Z_df)):
            row = Z_df.iloc[i]
            if row['Z'] < 0.3 and row['S'] < 3.5:
                lock_count += 1
                # Only fire when we've held the lock for exactly min_consecutive states
                if lock_count == min_consecutive:
                    events.append({
                        'Time'     : row['Time'],
                        'Epicenter': row['epi'],
                        'Outlet'   : row['out'],
                        'Z'        : row['Z'],
                        'S'        : row['S'],
                        'Lock_dur' : lock_count,
                    })
            else:
                lock_count = 0

        if not events:
            results[min_consecutive] = {'events': 0, 'win_rate': 0, 'ev': 0, 'exp': 0}
            continue

        ev_df    = pd.DataFrame(events)
        ev_df['HourGroup'] = ev_df['Time'].dt.floor('H')
        distinct = ev_df.drop_duplicates(subset=['HourGroup', 'Outlet'])

        passes, failures = 0, 0
        release_log, loss_log = [], []

        for _, event in distinct.iterrows():
            t0     = event['Time']
            outlet = event['Outlet']

            post = raw_df[raw_df.index >= t0].copy()
            if len(post) < 5:
                continue
            post['post_energy'] = post[vol_cols].sum(axis=1).cumsum()
            forward_df = post[post['post_energy'] <= 100000]
            if len(forward_df) < 5:
                continue

            p0   = forward_df[f"{outlet}_M1_close"].iloc[0]
            disp = np.log(forward_df[f"{outlet}_M1_close"] / p0) * 100

            max_up   = disp.max()
            max_down = abs(disp.min())
            release  = max(max_up, max_down)
            friction = min(max_up, max_down)

            if release > (friction * 2):
                passes += 1
                release_log.append(release)
            else:
                failures += 1
                loss_log.append(friction)

        total    = passes + failures
        win_rate = passes / total * 100 if total > 0 else 0
        avg_rel  = np.mean(release_log) if release_log else 0.0
        avg_loss = np.mean(loss_log)    if loss_log    else 0.0
        exp_r    = avg_rel / (avg_loss + 0.0001) if passes > 0 else 0.0
        ev       = (win_rate/100 * avg_rel) - ((1 - win_rate/100) * avg_loss)

        results[min_consecutive] = {
            'events'  : total,
            'passes'  : passes,
            'win_rate': win_rate,
            'avg_rel' : avg_rel,
            'avg_loss': avg_loss,
            'exp'     : exp_r,
            'ev'      : ev,
        }

    # --- REPORT ---
    print("\n" + "="*80)
    print("PERSISTENT Z-LOCK: PRESSURE BUILD-UP ANALYSIS")
    print("="*80)
    print(f"{'Lock States':>12} | {'Events':>6} | {'Win%':>7} | {'Avg Rel':>8} | {'Avg Loss':>9} | {'Exp':>6} | {'EV':>10}")
    print("-"*80)
    print(f"{'Baseline (1)':>12} | {'32':>6} | {'18.75%':>7} | {'0.6378%':>8} | {'0.0146%':>9} | {'43.32x':>6} | {'+0.10770%':>10}")
    for k, r in results.items():
        if r['events'] == 0:
            print(f"  {k} states  | {'0':>6} | {'---':>7} | {'---':>8} | {'---':>9} | {'---':>6} | {'---':>10}")
        else:
            ev_str = f"{r['ev']:+.5f}%"
            print(f"  {k} states  | {r['events']:>6} | {r['win_rate']:>6.2f}% | {r['avg_rel']:>7.4f}% | {r['avg_loss']:>8.4f}% | {r['exp']:>5.2f}x | {ev_str:>10}")
    print("="*80)
    print("\n[ FORENSIC: What do the winning events have in common? ]")

    # Re-examine the 1-state events with forensic details
    best = results.get(1, {})
    if best.get('events', 0) > 0:
        ev_df = pd.DataFrame(Z_series)
        lock_events = ev_df[(ev_df['Z'] < 0.3) & (ev_df['S'] < 3.5)]
        lock_events = lock_events.drop_duplicates(subset=['out'])
        print(f"  Unique outlets that triggered: {lock_events['out'].value_counts().to_dict()}")
        print(f"  Unique epicenters:             {lock_events['epi'].value_counts().to_dict()}")

if __name__ == "__main__":
    run_persistent_lock()
