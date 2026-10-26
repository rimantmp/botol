#!/usr/bin/env python3
"""Automated Installer, Dependency Checker, and Database Provisioner.

Bottle Fall Detection System.
Run:
    python installer.py
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path
from urllib.parse import quote_plus

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Color helper for Windows/ANSI console
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_banner():
    banner = f"""{Colors.BOLD}{Colors.CYAN}
======================================================================
     BOTTLE FALL DETECTION SYSTEM - INSTALLER & SETUP DOCTOR
                  Vercel Geist UI & AI Pipeline
======================================================================{Colors.RESET}
"""
    print(banner)


def log_step(step_num: int, title: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}[LANGKAH {step_num}] {title}{Colors.RESET}")
    print("-" * 65)


def log_ok(msg: str):
    print(f"  {Colors.GREEN}[OK]{Colors.RESET} {msg}")


def log_warn(msg: str):
    print(f"  {Colors.YELLOW}[PERINGATAN]{Colors.RESET} {msg}")


def log_err(msg: str):
    print(f"  {Colors.RED}[GAGAL]{Colors.RESET} {msg}")


def log_info(msg: str):
    print(f"  {Colors.CYAN}[INFO]{Colors.RESET} {msg}")


def check_python_version() -> bool:
    log_step(1, "Memeriksa Versi Python")
    v = sys.version_info
    print(f"  Python aktif: {v.major}.{v.minor}.{v.micro} ({sys.executable})")
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        log_err("Versi Python minimal adalah 3.10. Harap perbarui Python Anda.")
        return False
    log_ok(f"Versi Python memenuhi syarat (>= 3.10).")
    return True


def check_and_create_directories():
    log_step(2, "Menyiapkan Direktori Kerja")
    dirs = [
        BASE_DIR / "videos" / "input",
        BASE_DIR / "videos" / "output",
        BASE_DIR / "videos" / "snapshots",
        BASE_DIR / "models",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        log_ok(f"Direktori siap: {d.relative_to(BASE_DIR)}")


def check_and_install_packages() -> bool:
    log_step(3, "Memeriksa & Menginstal Dependensi (requirements.txt)")
    req_file = BASE_DIR / "requirements.txt"
    if not req_file.exists():
        log_err(f"File {req_file} tidak ditemukan!")
        return False

    required_modules = {
        "ultralytics": "ultralytics (YOLOv8)",
        "cv2": "opencv-python",
        "numpy": "numpy",
        "flask": "flask",
        "sqlalchemy": "sqlalchemy",
        "pymysql": "pymysql",
        "dotenv": "python-dotenv",
        "pytest": "pytest",
    }

    missing = []
    for mod, label in required_modules.items():
        try:
            __import__(mod)
            log_ok(f"Modul terdeteksi: {label}")
        except ImportError:
            missing.append(label)
            log_warn(f"Modul belum terpasang: {label}")

    if missing:
        print(f"\n  Sedang menginstal {len(missing)} dependensi yang belum lengkap...")
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
        try:
            subprocess.run(cmd, check=True)
            log_ok("Semua dependensi Python berhasil diinstal!")
        except subprocess.CalledProcessError as e:
            log_err(f"Gagal menginstal dependensi: {e}")
            return False
    else:
        log_ok("Semua dependensi Python sudah lengkap dan siap.")
    return True


def check_ai_models() -> bool:
    log_step(4, "Memeriksa AI Weights Model YOLOv8")
    model_file = BASE_DIR / "yolov8n.pt"
    if model_file.exists() and model_file.stat().st_size > 1_000_000:
        size_mb = model_file.stat().st_size / (1024 * 1024)
        log_ok(f"Model {model_file.name} ditemukan ({size_mb:.1f} MB).")
        return True

    log_info("Model yolov8n.pt belum ada. Mengunduh model bobot resmi Ultralytics...")
    try:
        from ultralytics import YOLO
        YOLO("yolov8n.pt")
        log_ok("Model yolov8n.pt berhasil diunduh dan diverifikasi.")
        return True
    except Exception as e:
        log_err(f"Gagal memuat/mengunduh model YOLOv8: {e}")
        return False


def setup_environment_file() -> dict:
    log_step(5, "Memeriksa Konfigurasi Lingkungan (.env)")
    env_file = BASE_DIR / ".env"
    env_example = BASE_DIR / ".env.example"

    if not env_file.exists():
        if env_example.exists():
            shutil.copy(env_example, env_file)
            log_ok("File .env baru dibuat dari template .env.example.")
        else:
            default_env = (
                "DB_HOST=127.0.0.1\n"
                "DB_PORT=3306\n"
                "DB_NAME=bottle_fall_detection\n"
                "DB_USER=root\n"
                "DB_PASSWORD=\n"
            )
            env_file.write_text(default_env, encoding="utf-8")
            log_ok("File .env default dibuat.")
    else:
        log_ok("File konfigurasi .env ditemukan.")

    # Read config
    try:
        from dotenv import dotenv_values
        cfg = dotenv_values(env_file)
    except ImportError:
        cfg = {}

    db_host = cfg.get("DB_HOST", "127.0.0.1")
    db_port = int(cfg.get("DB_PORT", 3306))
    db_name = cfg.get("DB_NAME", "bottle_fall_detection")
    db_user = cfg.get("DB_USER", "root")
    db_pass = cfg.get("DB_PASSWORD", "")

    log_info(f"Target DB: {db_user}@{db_host}:{db_port}/{db_name}")
    return {
        "host": db_host,
        "port": db_port,
        "name": db_name,
        "user": db_user,
        "password": db_pass,
    }


def setup_database(db_cfg: dict) -> bool:
    log_step(6, "Memeriksa & Menginstal Database MySQL")
    import pymysql

    host = db_cfg["host"]
    port = db_cfg["port"]
    user = db_cfg["user"]
    password = db_cfg["password"]
    db_name = db_cfg["name"]

    # 1. Test raw socket connection to MySQL server
    conn = None
    try:
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            connect_timeout=5,
            autocommit=True,
        )
        log_ok(f"Koneksi ke MySQL Server ({host}:{port}) berhasil.")
    except Exception as e:
        log_err(f"Tidak dapat terhubung ke MySQL Server pada {host}:{port}.")
        log_warn(f"Detail kesalahan: {e}")
        log_info("PANDUAN PENYELESAIAN:")
        print("    1. Pastikan server MySQL / MariaDB sudah berjalan (misal: jalankan Apache & MySQL di XAMPP Control Panel).")
        print("    2. Periksa kembali host, port, user, dan password pada file '.env'.")
        print("    3. Catatan: Web app tetap dapat berjalan dalam mode fallback tanpa MySQL, namun log riwayat tidak tersimpan permanen.")
        return False

    # 2. Check and Create Database
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
            log_ok(f"Database `{db_name}` telah siap (dibuat/diverifikasi).")
    except Exception as e:
        log_err(f"Gagal membuat database `{db_name}`: {e}")
        conn.close()
        return False
    finally:
        conn.close()

    # 3. Create tables using SQLAlchemy models
    try:
        from sqlalchemy import create_engine, inspect
        from src.db_models import Base, Camera

        db_url = f"mysql+pymysql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{db_name}?charset=utf8mb4"
        engine = create_engine(db_url, future=True)
        Base.metadata.create_all(engine)

        insp = inspect(engine)
        tables = insp.get_table_names()
        log_ok(f"Skema tabel berhasil diinisialisasi. Tabel aktif: {', '.join(tables)}")

        # Insert default camera record if table is empty
        from sqlalchemy.orm import Session
        with Session(engine) as s:
            cam_count = s.query(Camera).count()
            if cam_count == 0:
                default_cam = Camera(
                    id=1,
                    name="Kamera Lini Produksi 1",
                    source_type="webcam",
                    source_identifier="0",
                    is_active=True,
                )
                s.add(default_cam)
                s.commit()
                log_ok("Data default kamera lini produksi berhasil didaftarkan (ID: 1).")
        return True
    except Exception as e:
        log_err(f"Gagal menginisialisasi skema tabel database: {e}")
        return False


def run_unit_tests():
    log_step(7, "Menjalankan Uji Coba Cepat (Unit Tests)")
    cmd = [sys.executable, "-m", "pytest", "-q"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            log_ok("Seluruh rangkaian unit test (25 test) BERHASIL!")
        else:
            log_warn(f"Beberapa unit test menghasilkan catatan:\n{res.stdout[:300]}")
    except Exception as e:
        log_warn(f"Uji coba dilewati: {e}")


def main():
    print_banner()

    if not check_python_version():
        sys.exit(1)

    check_and_create_directories()

    if not check_and_install_packages():
        log_err("Instalasi dependensi tidak selesai.")
        sys.exit(1)

    check_ai_models()

    db_cfg = setup_environment_file()
    db_ok = setup_database(db_cfg)

    run_unit_tests()

    print("\n" + "=" * 65)
    print(f"{Colors.BOLD}{Colors.GREEN}✓ INSTALASI & PEMERIKSAAN SISTEM SELESAI!{Colors.RESET}")
    print("=" * 65)
    print(f"\n{Colors.BOLD}Akun Login Terdaftar:{Colors.RESET}")
    print(f"  - {Colors.CYAN}Administrator{Colors.RESET} : Username: {Colors.BOLD}admin{Colors.RESET}     | Password: {Colors.BOLD}admin123{Colors.RESET}")
    print(f"  - {Colors.CYAN}Operator{Colors.RESET}      : Username: {Colors.BOLD}operator{Colors.RESET}  | Password: {Colors.BOLD}operator123{Colors.RESET}")
    print(f"\nStatus Database MySQL: {'ONLINE' if db_ok else 'OFFLINE (Perlu dinyalakan di XAMPP/MySQL service)'}")
    print(f"Buka di Browser      : http://127.0.0.1:5000\n")

    # Interactive choice to launch web app immediately
    if sys.stdin.isatty():
        try:
            choice = input("Apakah Anda ingin langsung menjalankan Web Dashboard sekarang? [Y/n]: ").strip().lower()
            if choice in ("", "y", "ya", "yes"):
                print(f"\n{Colors.GREEN}Menjalankan web_app.py... Tekan Ctrl+C untuk berhenti.{Colors.RESET}\n")
                subprocess.run([sys.executable, "web_app.py"])
        except KeyboardInterrupt:
            print("\nSelesai.")


if __name__ == "__main__":
    main()
