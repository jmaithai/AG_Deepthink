# AION Prime - Disaster Recovery & Startup Sequence

If the system crashes, restarts, or you simply need to boot it from a cold start, follow this sequence exactly.

## 1. Prime the MT5 Terminal Memory (M1 History)
Before running the daemons, you **must** cache the deep M1 historical data into the terminal's RAM to build the 32-D Kinetic Layering. You do not need to press "Page Up" manually.

Run the auto-cache script:
```bash
cd e:\spiderweb_core\AG_Deepthink
python auto_cache_history.py
```
*(If you are running multiple terminals, you can pass `--config <path_to_config>` to target a specific terminal path).*

## 2. Boot the Dashboard Server (Optional but Recommended)
To visualize the structural tension and live trades:
```bash
cd e:\spiderweb_core\AG_Deepthink\dashboard
python dashboard_server.py
```
*(The dashboard now supports dynamic ports via `--config` so you can run multiple instances).*

## 3. Boot the Live Sentinel (AION Prime)
Once the terminal memory is primed, launch the daemon:
```bash
cd e:\spiderweb_core\AG_Deepthink
python aion_farm_daemon.py
```

### Safety Switch Override
The daemon defaults to `DRY_RUN = True` to prevent accidental live execution. 
To switch to live ammunition:
1. Open `aion_farm_daemon.py`
2. Change `DRY_RUN = False` (Line 21)
3. Restart the script.
