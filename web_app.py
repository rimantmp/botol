"""Flask web dashboard for Bottle Fall Detection System.

Run:
    python web_app.py
Then open http://127.0.0.1:5000

Features:
- Live MJPEG stream of the annotated pipeline output.
- Start/stop detection on webcam or recorded video.
- Live stats: FPS, objects, statuses, DB status.
- Event history from MySQL with filtering.
- Snapshot images referenced by events.
"""
import logging
import os
import time
import uuid
from datetime import datetime, timedelta

import cv2
import numpy as np
from flask import Flask, Response, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.utils import secure_filename

from config import AppConfig
from src.config_loader import load_config
from src.database import create_db_engine, check_connection, init_schema, get_session
from src.db_models import Camera, FallEvent
from src.web_pipeline import DetectionWorker

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("web_app")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "bottle-fall-vision-secret-key-2026")
cfg: AppConfig = load_config()
worker = DetectionWorker(cfg)
# One shared read engine for history queries (never created per request).
history_engine = create_db_engine(cfg.db)

ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload cap

# Built-in User Accounts for RBAC (Admin vs Operator)
USERS = {
    "admin": {
        "password": "admin123",
        "role": "ADMIN",
        "name": "System Administrator",
        "initials": "AD",
    },
    "operator": {
        "password": "operator123",
        "role": "OPERATOR",
        "name": "Operator Lini 1",
        "initials": "OP",
    },
}


def _init_db() -> bool:
    if check_connection(history_engine):
        init_schema(history_engine)
        return True
    return False


def _placeholder_jpeg(text: str) -> bytes:
    img = np.zeros((360, 640, 3), np.uint8)
    cv2.putText(img, text, (110, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (200, 200, 200), 2)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else b""


# ------------------------------------------------------------------ auth gate & routes

@app.before_request
def require_authentication():
    """Ensure every page and API route requires login before access."""
    open_endpoints = {"login", "logout", "api_auth_login", "static"}
    if request.endpoint and request.endpoint in open_endpoints:
        return None
    if request.path.startswith("/static"):
        return None

    if "user" not in session:
        if request.path.startswith("/api/") or request.path.startswith("/video_feed"):
            return jsonify({
                "ok": False,
                "message": "Autentikasi diperlukan. Silakan login terlebih dahulu."
            }), 401
        return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user" in session:
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/api/auth/login", methods=["POST"])
def api_auth_login():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip().lower()
    password = str(data.get("password", "")).strip()

    user_entry = USERS.get(username)
    if not user_entry or user_entry["password"] != password:
        return jsonify({"ok": False, "message": "Nama pengguna atau kata sandi salah."}), 401

    session["user"] = {
        "username": username,
        "name": user_entry["name"],
        "role": user_entry["role"],
        "initials": user_entry["initials"],
    }
    return jsonify({"ok": True, "user": session["user"]})


@app.route("/api/auth/me")
def api_auth_me():
    if "user" not in session:
        return jsonify({"ok": False, "user": None}), 401
    return jsonify({"ok": True, "user": session["user"]})


# ------------------------------------------------------------------ routes

@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", user=session.get("user"), db_ok=_init_db())


@app.route("/api/status")
def api_status():
    return jsonify(worker.get_stats())


@app.route("/api/log")
def api_log():
    return jsonify({"ok": True, "lines": worker.get_log()})


@app.route("/api/results")
def api_results():
    """Session results: summary + live event list from the running worker."""
    s = worker.get_stats()
    fall_events = [e for e in s["last_events"] if e["event_type"] == "BOTOL_JATUH"]
    recover_events = [e for e in s["last_events"] if e["event_type"] == "BOTOL_KEMBALI_BERDIRI"]
    return jsonify({
        "ok": True,
        "running": s["running"],
        "source": s["source"],
        "frames": s["frames"],
        "fps": s["fps"],
        "db_status": s["db_status"],
        "error": s["error"],
        "summary": {
            "total_events": s["events_this_session"],
            "falls": len(fall_events),
            "recoveries": len(recover_events),
        },
        "events": s["last_events"],
    })


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    """Get or update fall detection and model parameters (Admin calibration)."""
    if request.method == "POST":
        if session.get("user", {}).get("role") != "ADMIN":
            return jsonify({"ok": False, "message": "Akses ditolak: Hanya Administrator yang berwenang mengubah konfigurasi."}), 403
        data = request.get_json(silent=True) or {}
        if "virtual_line_y" in data:
            val = data["virtual_line_y"]
            cfg.fall.virtual_line_y = float(val) if val not in (None, "") else None
        if "min_downward_speed_px" in data and data["min_downward_speed_px"]:
            cfg.fall.min_downward_speed_px = float(data["min_downward_speed_px"])
        if "min_downward_drop_px" in data and data["min_downward_drop_px"]:
            cfg.fall.min_downward_drop_px = float(data["min_downward_drop_px"])
        if "fall_ratio" in data and data["fall_ratio"]:
            cfg.fall.fall_ratio = float(data["fall_ratio"])
        if "stand_ratio" in data and data["stand_ratio"]:
            cfg.fall.stand_ratio = float(data["stand_ratio"])
        if "conf_threshold" in data and data["conf_threshold"]:
            cfg.detection.conf_threshold = float(data["conf_threshold"])
        if "model_path" in data and data["model_path"]:
            cfg.detection.model_path = str(data["model_path"])
        return jsonify({"ok": True, "message": "Konfigurasi kalibrasi berhasil disimpan."})

    return jsonify({
        "ok": True,
        "config": {
            "virtual_line_y": cfg.fall.virtual_line_y,
            "min_downward_speed_px": cfg.fall.min_downward_speed_px,
            "min_downward_drop_px": cfg.fall.min_downward_drop_px,
            "fall_ratio": cfg.fall.fall_ratio,
            "stand_ratio": cfg.fall.stand_ratio,
            "confirm_frames": cfg.fall.confirm_frames,
            "conf_threshold": cfg.detection.conf_threshold,
            "model_path": cfg.detection.model_path,
        }
    })


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(silent=True) or {}
    source_kind = data.get("source", "webcam")
    if source_kind == "webcam":
        source = int(data.get("camera_index", 0))
    else:
        path = str(data.get("video_path", "")).strip()
        if not path:
            return jsonify({"ok": False, "message": "video_path diperlukan."}), 400
        # Check if file exists as-is, or in input_dir, or in root
        if not os.path.exists(path):
            if os.path.exists(os.path.join(cfg.input_dir, path)):
                path = os.path.join(cfg.input_dir, path)
            elif os.path.exists(os.path.basename(path)):
                path = os.path.basename(path)
        source = path
    # Update virtual line if passed
    if "virtual_line_y" in data and data["virtual_line_y"] is not None:
        try:
            cfg.fall.virtual_line_y = float(data["virtual_line_y"])
        except (ValueError, TypeError):
            pass
    ok, message = worker.start(source, camera_name=cfg.camera_name)
    return jsonify({"ok": ok, "message": message}), (200 if ok else 409)


@app.route("/api/stop", methods=["POST"])
def api_stop():
    worker.stop()
    return jsonify({"ok": True, "message": "Dihentikan."})


@app.route("/video_feed")
def video_feed():
    def generate():
        # Runs until the browser disconnects (GeneratorExit); serves the last
        # frame while idle so the <img> keeps showing recent output.
        while True:
            jpeg = worker.get_jpeg()
            if jpeg is None:
                frame_data = _placeholder_jpeg("Menunggu stream...")
            else:
                frame_data = jpeg
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n")
            # ~15 fps delivery when live; slow heartbeat when idle.
            time.sleep(0.066 if jpeg is not None else 0.5)

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/events")
def api_events():
    if not _init_db():
        return jsonify({"ok": False, "message": "Database offline."}), 503

    event_type = request.args.get("event_type", "").strip()
    days = request.args.get("days", type=int)

    from sqlalchemy import select
    stmt = select(FallEvent).order_by(FallEvent.occurred_at.desc()).limit(200)
    if event_type:
        stmt = stmt.where(FallEvent.event_type == event_type)
    if days:
        since = datetime.utcnow() - timedelta(days=days)
        stmt = stmt.where(FallEvent.occurred_at >= since)

    with get_session(history_engine) as session:
        rows = session.scalars(stmt).all()
        events = [{
            "id": r.id, "camera_id": r.camera_id, "track_id": r.track_id,
            "event_type": r.event_type, "confidence": r.confidence,
            "orientation_ratio": r.orientation_ratio,
            "occurred_at": r.occurred_at.strftime("%Y-%m-%d %H:%M:%S"),
            "snapshot_path": r.snapshot_path,
            "snapshot_url": ("/snapshots/" + os.path.basename(r.snapshot_path))
                            if r.snapshot_path else None,
        } for r in rows]
    return jsonify({"ok": True, "events": events})


@app.route("/snapshots/<path:filename>")
def snapshots(filename):
    return send_from_directory(cfg.snapshot_dir, filename)


@app.route("/api/cameras")
def api_cameras():
    if not _init_db():
        return jsonify({"ok": False, "message": "Database offline."}), 503
    from sqlalchemy import select
    with get_session(history_engine) as session:
        cams = session.scalars(select(Camera).order_by(Camera.id)).all()
        return jsonify({"ok": True, "cameras": [
            {"id": c.id, "name": c.name, "source_type": c.source_type,
             "source_identifier": c.source_identifier} for c in cams]})


# ------------------------------------------------------------- video files

@app.route("/api/videos", methods=["GET"])
def api_videos():
    """List uploadable video files in input directory and root workspace."""
    files = []
    seen = set()
    for directory in [cfg.input_dir, "."]:
        if not os.path.exists(directory):
            continue
        try:
            for name in sorted(os.listdir(directory)):
                if name.startswith("."):
                    continue
                path = os.path.join(directory, name)
                if os.path.isfile(path) and os.path.splitext(name)[1].lower() in ALLOWED_VIDEO_EXT:
                    if name not in seen:
                        seen.add(name)
                        files.append({
                            "name": name,
                            "path": path.replace("\\", "/"),
                            "size_mb": round(os.path.getsize(path) / 1e6, 1)
                        })
        except OSError:
            pass
    return jsonify({"ok": True, "videos": files})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload a video file into videos/input/ for detection."""
    if "file" not in request.files:
        return jsonify({"ok": False, "message": "Field 'file' tidak ada."}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"ok": False, "message": "Nama file kosong."}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXT:
        return jsonify({"ok": False,
                        "message": f"Ekstensi {ext} tidak didukung. Gunakan: "
                                   f"{', '.join(sorted(ALLOWED_VIDEO_EXT))}"}), 400

    os.makedirs(cfg.input_dir, exist_ok=True)
    # Prefix with short uuid so a re-upload never overwrites an existing file.
    stored = f"{uuid.uuid4().hex[:8]}_{secure_filename(f.filename)}"
    dest = os.path.join(cfg.input_dir, stored)
    f.save(dest)
    size_mb = round(os.path.getsize(dest) / 1e6, 1)
    logger.info("Video uploaded: %s (%.1f MB)", stored, size_mb)
    return jsonify({"ok": True, "filename": stored, "size_mb": size_mb})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)
