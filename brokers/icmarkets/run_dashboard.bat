@echo off
title AION Dashboard - ICMarkets
echo [*] Booting Dashboard Server (ICMarkets)
python -X utf8 ../../dashboard/dashboard_server.py --config config.json
pause
