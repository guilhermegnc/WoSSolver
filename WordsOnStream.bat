@echo off
cd /d "%~dp0"
start "" "%~dp0\venv\Scripts\python.exe" "%~dp0\solver.py"
exit
