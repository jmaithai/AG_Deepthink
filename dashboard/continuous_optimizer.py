import time
import subprocess
import os

print("[*] Starting Continuous Sovereign Calibration Daemon...")
print("[*] Engine will recalibrate 38-D fluid parameters every 60 minutes.")

while True:
    print(f"\n[{time.strftime('%H:%M:%S')}] Triggering Continuous Calibration Loop...")
    
    # Run the optimizer (using the historical data subset)
    try:
        print("[*] Spawning Optimizer Subprocess...")
        subprocess.run(["python", "optimizer_38d.py"], check=True)
        print("[+] Optimization complete. Tensor updated.")
    except Exception as e:
        print(f"[-] Optimizer failed: {e}")
        
    print("[*] Sleeping for 60 minutes...")
    time.sleep(3600)
