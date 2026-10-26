@echo off
chcp 65001 > nul
title Bottle Fall Detection System - Console Dashboard
color 0A

echo ======================================================================
echo       BOTTLE FALL DETECTION SYSTEM - CONSOLE DASHBOARD
echo ======================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual Environment .venv belum ada!
    echo Harap jalankan 'setup.bat' terlebih dahulu untuk instalasi otomatis.
    echo.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo Membuka browser ke http://127.0.0.1:5000...
start http://127.0.0.1:5000

echo Menjalankan Flask Web Console...
echo Tekan Ctrl+C di jendela ini untuk mematikan server.
echo.
python web_app.py

pause
