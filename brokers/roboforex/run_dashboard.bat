@echo off
title AION Dashboard - Roboforex
echo [*] Booting Dashboard Server (Roboforex)
python -X utf8 ../../dashboard/dashboard_server.py --config config.json
pause
