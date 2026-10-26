# Bottle Fall Detection System

Sistem deteksi botol jatuh berbasis computer vision: deteksi botol (YOLO),
tracking antar-frame, analisis orientasi + temporal confirmation, pencatatan
kejadian ke MySQL, dan visualisasi real-time di layar.

Berjalan di **Windows + CPU** (tanpa GPU/CUDA wajib).

---

## 1. Persiapan

### Python

Butuh Python 3.11+ (diuji dengan 3.12.10).

### Instalasi Otomatis (One-Click)

Di Windows, Anda cukup menjalankan:
- **`setup.bat`**: Otomatis mengecek Python, membuat `.venv`, menginstal dependensi (`requirements.txt`), memeriksa model YOLOv8, mengecek & membuat database MySQL beserta tabelnya, dan menjalankan unit test.
- Atau jalankan via terminal:
```powershell
python installer.py
```

### Menjalankan Web Console (One-Click)
- **`run.bat`**: Otomatis mengaktifkan environment `.venv`, membuka browser ke `http://127.0.0.1:5000`, dan menjalankan server Flask.

### Kredensial Login Bawaan:
- **Admin**: `admin` / `admin123` (Hak akses penuh: kalibrasi AI, tripwire ketinggian, upload video, RBAC)
- **Operator**: `operator` / `operator123` (Hak akses operasional: live monitoring, log riwayat)

### 2. Model & konfigurasi

| Variabel | Default | Arti |
|---|---|---|
| `DB_HOST` | `127.0.0.1` | Host MySQL |
| `DB_PORT` | 3306 | Port MySQL |
| `DB_NAME` | bottle_fall_detection | Nama database |
| `DB_USER` | bottle_app | User database |
| `DB_PASSWORD` | — | Password (jangan di-commit) |
| `DB_URL` | — | Override URL penuh (mis. untuk testing) |
| `YOLO_MODEL_PATH` | `yolov8n.pt` | Model YOLO (pralatih atau custom) |
| `CONF_THRESHOLD` | 0.35 | Ambang confidence deteksi |
| `IMGSZ` | 640 | Ukuran input YOLO (480 lebih cepat, lebih sedikit akurat) |
| `CAMERA_NAME` | default | Nama kamera untuk tabel cameras |
| `FALL_RATIO` | 0.95 | ratio (h/w) ≤ nilai → kandidat jatuh |
| `STAND_RATIO` | 1.55 | ratio ≥ nilai → kandidat berdiri |
| `CONFIRM_FRAMES` | 3 | Frame beruntun untuk konfirmasi transisi |

### 3. Menjalankan

Webcam:
```powershell
.\.venv\Scripts\python main.py --source webcam
```

Video rekaman:
```powershell
.\.venv\Scripts\python main.py --source video --video-path videos/input/nama.mp4
```

Tanpa DB / tanpa jendela:
```powershell
.\.venv\Scripts\python main.py --source video --video-path x.mp4 --no-db --no-display
```

Konteks kode: bounding box, ID tracking, status, FPS, jumlah objek, status DB
ditampilkan di jendela OpenCV. Tekan `q` atau `ESC` untuk keluar.

### 3b. GUI berbasis web (Flask)

```powershell
.\.venv\Scripts\python web_app.py
```

Buka `http://127.0.0.1:5000` di browser. Fitur:

- **Stream live** MJPEG dari pipeline deteksi (webcam atau video file).
- **Mulai/Hentikan** deteksi dari browser — pilih webcam (indeks) atau file video.
- **Status live**: FPS, jumlah objek, frame diproses, status per track, status DB.
- **Riwayat kejadian** dari MySQL dengan filter tipe (`BOTOL_JATUH` /
  `BOTOL_KEMBALI_BERDIRI`) dan periode (24 jam / 7 hari).
- Snapshot kejadian tersedia via `/snapshots/<file>` bila fitur aktif.

Catatan: pipeline berjalan di thread background di mesin yang sama; browser
hanya menerima hasil anotasi. Satu sumber deteksi aktif pada satu waktu
(ganti sumber = hentikan dulu).

### 4. Riwayat kejadian

```sql
USE bottle_fall_detection;
SELECT id, track_id, event_type, confidence, occurred_at, snapshot_path
FROM fall_events ORDER BY occurred_at DESC LIMIT 50;
```

### 5. Struktur modul

| File | Fungsi |
|---|---|
| `main.py` | Entry point + pipeline utama |
| `config.py` | Dataclass konfigurasi dengan default |
| `src/config_loader.py` | Baca `.env` + override CLI |
| `src/video_source.py` | Webcam/video rekaman, reconnect |
| `src/detector.py` | Deteksi botol (YOLO) |
| `src/tracker.py` | IoU tracker + ByteTrack facade |
| `src/fall_detector.py` | Status machine BERDIRI/JATUH/TIDAK_PASTI |
| `src/event_logger.py` | Simpan event + snapshot ke MySQL |
| `src/database.py` | Engine, session, schema init, connection check |
| `src/db_models.py` | ORM: cameras, detections, fall_events |
| `src/visualizer.py` | Anotasi + HUD di frame |
| `src/web_pipeline.py` | Pipeline deteksi di background thread utk web UI |
| `web_app.py` | Flask dashboard: MJPEG stream, API, riwayat |
| `templates/index.html` | Halaman dashboard |
| `tests/` | 22 test: fall logic, event, database, video source |

### 6. Keterbatasan (jujur)

- **Model pralatih COCO** hanya kelas `bottle` (39). Botol kaca/transparan,
  ukuran kecil, atau bentuk tidak umum sering miss — fine-tuning mungkin
  diperlukan (lihat prompt.md §4).
- **Rasio bbox bukan sudut fisik**: perspektif, bentuk botol, oklusi mempengaruhi.
  Ambang default perlu dikalibrasi dengan video aktual dari kamera samping.
- **MySQL end-to-end sudah diuji** (Laragon, MySQL 8.0.30): koneksi, skema,
  insert event, read-back. Namun belum diuji di bawah beban/concurrent.
- **Belum diuji dengan video botol nyata** — akurasi/FPS pada footage asli
  belum diukur. Jangan anggap angka benchmark blank-frame sebagai performa
  real-world.
- **IoU tracker fallback** (bukan ByteTrack penuh) dipakai di v1; ByteTrack
  penuh perlu integrasi di main loop.
- Dashboard Streamlit belum dibuat — tahap lanjutan setelah fondasi stabil.

### 7. Testing

```powershell
.\.venv\Scripts\python -m pytest tests/ -v
```

22 test: fall logic (confirm frames, gray zone, no-repeat events, recovery),
event logging (store, failure → False, snapshot, no-duplicate), database
(schema, FK, rollback, connection failure), video source (missing file,
webcam gagal, release idempotent).

Batasan pengujian: tidak ada video botol nyata / webcam di environment ini,
jadi metrik precision/recall belum bisa diukur. Evaluasi lapangan diperlukan
(prompt.md §15).
