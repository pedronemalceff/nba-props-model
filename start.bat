@echo off
REM Double-click to start the NBA Props Model and open it in your browser.
cd /d "%~dp0backend"
start "" cmd /c "timeout /t 5 /nobreak >nul & start http://localhost:8000"
echo Starting NBA Props Model... leave this window open while you use it.
py -3.13 -m uvicorn nba_props.api:app --port 8000
pause
