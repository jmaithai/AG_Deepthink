@echo off
title AION Dashboard - Blackbull
echo [*] Booting Dashboard Server (Blackbull)
python -X utf8 ../../dashboard/dashboard_server.py --config config.json
pause
