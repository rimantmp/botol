@echo off
chcp 65001 > nul
title Bottle Fall Detection System - Setup ^& Installer
color 0B

echo ======================================================================
echo      BOTTLE FALL DETECTION SYSTEM - WINDOWS SETUP LAUNCHER
echo                 (Dengan Auto-Install Python)
echo ======================================================================
echo.

:: -------------------------------------------------------------------------
:: 1. Periksa Apakah Python Sudah Ada dan Memenuhi Syarat (>= 3.10)
:: -------------------------------------------------------------------------
set "PYTHON_EXEC="

:: Cek python standar di PATH
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_EXEC=python"
    goto :python_ready
)

:: Cek py launcher
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_EXEC=py -3"
    goto :python_ready
)

:: Cek path instalasi umum Python 3.12 jika belum terdaftar di PATH
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON_EXEC=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    set "PATH=%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%PATH%"
    goto :python_ready
)
if exist "%ProgramFiles%\Python312\python.exe" (
    set "PYTHON_EXEC=%ProgramFiles%\Python312\python.exe"
    set "PATH=%ProgramFiles%\Python312;%ProgramFiles%\Python312\Scripts;%PATH%"
    goto :python_ready
)

:: -------------------------------------------------------------------------
:: 2. Jika Python Belum Ada atau Versi < 3.10 -> Auto-Install Python 3.12
:: -------------------------------------------------------------------------
echo [PERINGATAN] Python 3.10 atau lebih baru belum terdeteksi.
echo [AUTO-SETUP] Memulai instalasi otomatis Python 3.12 resmi...
echo.

:: Opsi A: Coba gunakan winget (Windows Package Manager bawaan Windows 10/11)
where winget >nul 2>&1
if %errorlevel% equ 0 (
    echo [METODE 1] Menginstal Python 3.12 via Windows Package Manager (winget)...
    winget install --id Python.Python.3.12 -e --silent --accept-package-agreements --accept-source-agreements
    if %errorlevel% equ 0 goto :refresh_path
)

:: Opsi B: Unduh langsung installer resmi 64-bit dari python.org via PowerShell
echo [METODE 2] Mengunduh installer resmi Python 3.12 dari python.org...
set "PY_INSTALLER=%TEMP%\python-3.12.8-amd64.exe"
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe', '%PY_INSTALLER%')"

if exist "%PY_INSTALLER%" (
    echo [METODE 2] Menjalankan instalasi Python 3.12 secara silent (tanpa pop-up)...
    "%PY_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 SimpleInstall=1
    del "%PY_INSTALLER%" >nul 2>&1
) else (
    echo [ERROR] Gagal mengunduh installer Python. Pastikan koneksi internet aktif.
    pause
    exit /b 1
)

:refresh_path
:: Muat ulang variabel PATH dari Registry Windows agar terminal mengenali python
for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USER_PATH=%%b"
for /f "tokens=2*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "SYS_PATH=%%b"
set "PATH=%USER_PATH%;%SYS_PATH%;%PATH%"

if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON_EXEC=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    set "PATH=%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%PATH%"
) else (
    set "PYTHON_EXEC=python"
)

:: Verifikasi ulang setelah auto-install
%PYTHON_EXEC% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Instalasi Python selesai, namun terminal membutuhkan restart.
    echo Harap tutup jendela ini dan jalankan kembali setup.bat.
    pause
    exit /b 1
)
echo [OK] Python 3.12 berhasil diinstal dan siap digunakan!
echo.

:python_ready
echo [OK] Menggunakan Python runtime aktif.
echo.

:: -------------------------------------------------------------------------
:: 3. Buat Virtual Environment (.venv) Jika Belum Ada
:: -------------------------------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [SETUP] Membuat Virtual Environment venv...
    %PYTHON_EXEC% -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Gagal membuat virtual environment!
        pause
        exit /b 1
    )
    echo [OK] Virtual Environment berhasil dibuat.
    echo.
)

:: -------------------------------------------------------------------------
:: 4. Jalankan Installer ^& Database Provisioner di dalam .venv
:: -------------------------------------------------------------------------
echo [SETUP] Menjalankan installer, pengecekan dependensi, dan database...
echo.
call .venv\Scripts\activate.bat
python installer.py

if %errorlevel% neq 0 (
    echo.
    echo [CATATAN] Installer selesai dengan kode status: %errorlevel%
)

echo.
pause
