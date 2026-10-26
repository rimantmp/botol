"""Background detection worker for the Flask web UI.

Runs the same pipeline as main.py (video source -> YOLO -> tracking ->
fall logic -> MySQL event logging) in a thread, keeps the latest annotated
JPEG frame + stats for the web layer to serve. No OpenCV windows here.
"""
import logging
import threading
import time
from collections import deque
from typing import Optional

import cv2

from config import AppConfig
from src.database import create_db_engine, check_connection, init_schema, get_session
from src.detector import BottleDetector
from src.event_logger import EventLogger
from src.fall_detector import FallDetector
from src.tracker import Tracker
from src.video_source import VideoReadError, VideoSource
from src.visualizer import Visualizer

logger = logging.getLogger(__name__)


class DetectionWorker:
    """One shared detection pipeline driving the web dashboard."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        self._jpeg: Optional[bytes] = None
        self.stats = {
            "running": False,
            "source": "",
            "fps": 0.0,
            "objects": 0,
            "statuses": {},          # track_id -> status
            "db_status": "not started",
            "frames": 0,
            "events_this_session": 0,
            "last_events": [],       # recent transitions for the dashboard
            "error": "",
        }
        self._log: deque = deque(maxlen=200)   # live process log for the UI

    # ------------------------------------------------------------- control

    def start(self, source, camera_name: str = "default") -> tuple[bool, str]:
        """Start detection on a source. Fails if already running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False, "Sudah berjalan. Hentikan dulu sebelum ganti sumber."
            self._stop_evt.clear()
            self.stats.update(running=True, source=str(source), error="",
                              frames=0, events_this_session=0,
                              statuses={}, last_events=[])
            self._thread = threading.Thread(
                target=self._run, args=(source, camera_name), daemon=True)
            self._thread.start()
            return True, "OK"

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        with self._lock:
            self.stats["running"] = False

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -------------------------------------------------------------- output

    def log(self, msg: str) -> None:
        """Append a timestamped line to the live process log."""
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        with self._lock:
            self._log.append(line)

    def get_log(self) -> list:
        with self._lock:
            return list(self._log)

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self.stats)

    # ----------------------------------------------------------- internals

    def _set(self, **kwargs) -> None:
        with self._lock:
            self.stats.update(kwargs)

    def _run(self, source, camera_name: str) -> None:
        cfg = self.cfg
        self.log(f"Mulai deteksi: sumber {source!r}")
        db_status = "disabled"
        engine = None
        try:
            engine = create_db_engine(cfg.db)
            if check_connection(engine):
                init_schema(engine)
                db_status = "connected"
            else:
                db_status = "OFFLINE (events not stored)"
                engine = None
        except Exception as exc:
            logger.warning("DB init failed in worker: %s", exc.__class__.__name__)
            db_status = "OFFLINE (events not stored)"
            engine = None
        self.log(f"Database: {db_status}")
        self._set(db_status=db_status)

        video = VideoSource(source)
        try:
            video.open()
        except VideoReadError as exc:
            self.log(f"ERROR: {exc}")
            self._set(running=False, error=str(exc))
            return

        detector = BottleDetector(cfg.detection)
        try:
            detector.load()
        except Exception as exc:
            self.log(f"ERROR: gagal memuat model: {exc}")
            self._set(running=False, error=f"Gagal memuat model: {exc}")
            return

        self.log("Model YOLO dimuat.")
        tracker = Tracker()
        fall = FallDetector(cfg.fall)
        viz = Visualizer(source_label=str(source), virtual_line_y=cfg.fall.virtual_line_y)
        viz.db_status = db_status

        camera_id = None
        frame_idx = 0
        fps_prev = time.perf_counter()
        fps = 0.0

        try:
            while not self._stop_evt.is_set():
                ok, frame = video.read()
                if not ok or frame is None:
                    self.log("Sumber video selesai / terputus.")
                    self._set(error="Sumber video selesai / terputus.")
                    break

                detections = detector.detect(frame)
                tracks = tracker.update([d.bbox for d in detections], frame_idx, time.time())
                if frame_idx % 30 == 0:
                    self.log(f"frame {frame_idx}: {len(detections)} deteksi, "
                             f"{len(tracks)} track, {fps:.1f} fps")
                for tr in tracks:
                    for d in detections:
                        if d.bbox == tr.bbox:
                            tr.confidence = d.confidence

                transitions = fall.process_frame(tracks, frame_idx, time.time())
                statuses = {tr.track_id: fall.get_status(tr.track_id) for tr in tracks}

                annotated = viz.draw(frame, tracks, statuses, fps)

                if transitions and engine is not None:
                    if camera_id is None:
                        camera_id = self._resolve_camera_id(engine, cfg, source)
                    ev_logger = EventLogger(engine, camera_id=camera_id,
                                            snapshot_dir=cfg.snapshot_dir,
                                            save_snapshots=cfg.save_snapshots)
                    new_events = []
                    for tr_ev in transitions:
                        self.log(f"TRANSISI: track {tr_ev['track_id']} "
                                 f"{tr_ev['event_type']} "
                                 f"(conf {tr_ev['confidence']:.2f})")
                        if ev_logger.log_event(tr_ev, frame=annotated):
                            new_events.append({
                                "event_type": tr_ev["event_type"],
                                "track_id": tr_ev["track_id"],
                                "confidence": round(tr_ev["confidence"], 2),
                                "occurred_at": time.strftime(
                                    "%Y-%m-%d %H:%M:%S",
                                    time.localtime(tr_ev["occurred_at"])),
                            })
                    if new_events:
                        with self._lock:
                            lst = self.stats["last_events"]
                            self.stats["last_events"] = (new_events + lst)[:10]
                            self.stats["events_this_session"] += len(new_events)

                ok_enc, buf = cv2.imencode(".jpg", annotated,
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok_enc:
                    with self._lock:
                        self._jpeg = buf.tobytes()
                    self._set(objects=len(tracks), statuses=statuses)

                frame_idx += 1
                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - fps_prev, 1e-6))
                fps_prev = now
                self._set(frames=frame_idx, fps=round(fps, 1))
        finally:
            video.release()
            self.log(f"Selesai. Total frame: {frame_idx}.")
            self._set(running=False)

    @staticmethod
    def _resolve_camera_id(engine, cfg: AppConfig, source) -> int:
        from sqlalchemy import select
        from src.db_models import Camera

        source_type = "webcam" if isinstance(source, int) else "video"
        with get_session(engine) as session:
            cam = session.scalars(select(Camera).where(
                Camera.source_type == source_type,
                Camera.source_identifier == str(source))).first()
            if cam is None:
                cam = Camera(name=cfg.camera_name, source_type=source_type,
                             source_identifier=str(source))
                session.add(cam)
                session.flush()
            return cam.id
