@echo off
rem Starts the game and the webcam tracking together.
rem Stop by closing the game window or pressing Q in the webcam window.
cd /d "%~dp0Python"
"..\.venv\Scripts\python.exe" main.py --play %*
if errorlevel 1 pause
