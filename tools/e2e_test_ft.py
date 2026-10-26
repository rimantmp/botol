"""End-to-end test: fine-tuned model -> tracker -> fall logic on held-out v1."""
import sys
import time

sys.path.insert(0, ".")
import cv2
from ultralytics import YOLO

from config import FallConfig
from src.tracker import Tracker
from src.fall_detector import FallDetector


class FTDetector:
    """Class-agnostic wrapper: single-class fine-tuned model."""

    def __init__(self, path, conf=0.25):
        self.m = YOLO(path)
        self.conf = conf

    def detect_boxes(self, frame):
        r = self.m.predict(frame, conf=self.conf, verbose=False)[0]
        out = []
        if r.boxes is not None:
            for b in r.boxes:
                box = tuple(float(v) for v in b.xyxy[0].tolist())
                out.append((box, float(b.conf.item())))
        return out


def main():
    det = FTDetector("runs/detect/runs_ft/bottle_ft2/weights/best.pt")
    tr = Tracker()
    f = FallDetector(FallConfig())
    cap = cv2.VideoCapture("videos/input/video_01.mp4")
    i = 0
    events = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = det.detect_boxes(frame)
        tracks = tr.update([d[0] for d in dets], i, time.time())
        for t in tracks:
            t.confidence = next((c for b, c in dets if b == t.bbox), 0.5)
        trans = f.process_frame(tracks, i, time.time())
        events.extend(trans)
        st = {t.track_id: f.get_status(t.track_id) for t in tracks}
        if i % 10 == 0 or trans:
            line = f"f{i:3d}: tracks={len(tracks)} status={st}"
            if trans:
                line += " EVENT=" + str([x["event_type"] for x in trans])
            print(line)
        i += 1
    cap.release()
    print()
    print("EVENTS:", [(e["event_type"], "track", e["track_id"]) for e in events])


if __name__ == "__main__":
    main()
