"""Bottle Fall Detection System - main entry point.

Pipeline: video source -> YOLO detection -> tracking -> fall logic ->
event logging (MySQL) -> visualization.

Usage:
    python main.py --source webcam [--camera-index 0]
    python main.py --source video --video-path videos/input/demo.mp4
    python main.py --source video --video-path clip.mp4 --no-db --no-display
"""
import argparse
import logging
import sys
import time

import cv2

sys.path.insert(0, ".")  # allow `python main.py` from project root

from config import AppConfig
from src.config_loader import load_config
from src.database import check_connection, create_db_engine, get_session, init_schema
from src.detector import BottleDetector
from src.fall_detector import FallDetector
from src.tracker import Tracker
from src.video_source import VideoReadError, VideoSource
from src.visualizer import Visualizer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bottle Fall Detection System")
    p.add_argument("--source", choices=["webcam", "video"], default="webcam")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--video-path", type=str, default=None)
    p.add_argument("--no-db", action="store_true", help="run without database logging")
    p.add_argument("--no-display", action="store_true", help="headless run (no window)")
    p.add_argument("--save-output", action="store_true", help="save annotated video")
    p.add_argument("--conf", type=float, default=None, help="confidence threshold")
    p.add_argument("--imgsz", type=int, default=None, help="YOLO input size")
    p.add_argument("--virtual-line", type=float, default=None, help="Y coordinate for height drop line")
    return p.parse_args()


def resolve_source(args) -> object:
    if args.source == "webcam":
        return args.camera_index
    if not args.video_path:
        logger.error("--video-path is required with --source video")
        sys.exit(2)
    return args.video_path


def resolve_camera_id(engine, cfg: AppConfig, source_label: str) -> int:
    """Get-or-create the cameras row for this source. Returns its id."""
    from sqlalchemy import select
    from src.db_models import Camera

    source_type = "webcam" if isinstance(source_label, int) else "video"
    with get_session(engine) as session:
        cam = session.scalars(select(Camera).where(
            Camera.source_type == source_type,
            Camera.source_identifier == str(source_label))).first()
        if cam is None:
            cam = Camera(name=cfg.camera_name, source_type=source_type,
                         source_identifier=str(source_label))
            session.add(cam)
            session.flush()
        return cam.id


def main() -> int:
    args = parse_args()
    cfg: AppConfig = load_config(
        conf_threshold=args.conf, imgsz=args.imgsz,
        save_output_video=args.save_output, display=not args.no_display,
        fall_virtual_line_y=args.virtual_line,
    )

    # --- Database (single reusable engine, schema init, connectivity check) ---
    engine = None
    db_status = "disabled" if args.no_db else "connecting..."
    if not args.no_db:
        engine = create_db_engine(cfg.db)
        if check_connection(engine):
            init_schema(engine)
            db_status = "connected"
            logger.info("Database connected: %s@%s:%s/%s",
                        cfg.db.user, cfg.db.host, cfg.db.port, cfg.db.name)
        else:
            db_status = "OFFLINE (events not stored)"
            logger.warning("Database unavailable - continuing without event storage.")

    # --- Components ---
    source = resolve_source(args)
    video = VideoSource(source)
    try:
        video.open()
    except VideoReadError as exc:
        logger.error("%s", exc)
        return 1

    detector = BottleDetector(cfg.detection)
    detector.load()

    tracker = Tracker()
    fall = FallDetector(cfg.fall)
    viz = Visualizer(source_label=str(source), virtual_line_y=cfg.fall.virtual_line_y)
    viz.db_status = db_status

    writer = None
    if cfg.save_output_video:
        import os
        os.makedirs(cfg.output_dir, exist_ok=True)
        fps_in = video.cap.get(5) or 25.0
        w = int(video.cap.get(3)); h = int(video.cap.get(4))
        out_path = f"{cfg.output_dir}/out_{int(time.time())}.mp4"
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_in, (w, h))
        logger.info("Saving annotated video to %s", out_path)

    # --- Main loop ---
    frame_idx = 0
    camera_id = None  # resolved lazily on first event (avoids DB hit per frame)
    fps_prev = time.perf_counter()
    fps = 0.0
    try:
        while True:
            ok, frame = video.read()
            if not ok or frame is None:
                logger.info("Source ended or unavailable (frame %d).", frame_idx)
                break

            detections = detector.detect(frame)
            bbox_list = [d.bbox for d in detections]
            tracks = tracker.update(bbox_list, frame_idx, time.time())
            conf_by_track = {}
            for tr in tracks:
                # attach latest detection confidence for this track's bbox
                for d in detections:
                    if d.bbox == tr.bbox:
                        conf_by_track[tr.track_id] = d.confidence
                        tr.confidence = d.confidence

            transitions = fall.process_frame(tracks, frame_idx, time.time())
            statuses = {tr.track_id: fall.get_status(tr.track_id) for tr in tracks}

            for tr_ev in transitions:
                logger.info(
                    "EVENT: track_id=%s %s -> %s (event=%s, reason=%s, conf=%.2f)",
                    tr_ev["track_id"],
                    tr_ev["from_status"],
                    tr_ev["to_status"],
                    tr_ev["event_type"],
                    tr_ev.get("reason", "ORIENTATION"),
                    tr_ev.get("confidence", 0.0),
                )

            if transitions and engine is not None:
                if camera_id is None:
                    camera_id = resolve_camera_id(engine, cfg, source)
                from src.event_logger import EventLogger
                ev_logger = EventLogger(engine, camera_id=camera_id,
                                        snapshot_dir=cfg.snapshot_dir,
                                        save_snapshots=cfg.save_snapshots)
                for tr_ev in transitions:
                    stored = ev_logger.log_event(tr_ev, frame=frame)
                    if not stored:
                        viz.db_status = "DB WRITE FAILED"
                    # NOTE: camera_id=1 placeholder; cameras table managed in DB phase 2.

            if cfg.display or writer is not None:
                out_frame = viz.draw(frame, tracks, statuses, fps)
                if writer is not None:
                    writer.write(out_frame)
                if cfg.display:
                    cv2.imshow("Bottle Fall Detection", out_frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):  # ESC or q
                        logger.info("Exit requested by user.")
                        break

            frame_idx += 1
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - fps_prev, 1e-6))
            fps_prev = now
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        if writer is not None:
            writer.release()
        video.release()
        if cfg.display:
            cv2.destroyAllWindows()
        logger.info("Processed %d frames.", frame_idx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
