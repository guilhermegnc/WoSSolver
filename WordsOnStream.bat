@echo off
cd /d "%~dp0"
start "" "%~dp0\venv\Scripts\python.exe" "%~dp0\src\solver.py"
exit
