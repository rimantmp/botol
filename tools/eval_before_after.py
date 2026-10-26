"""Compare pretrained vs fine-tuned model on the held-out val video (v1).

Prints per-phase detection counts and visual comparison images.
"""
import sys

sys.path.insert(0, ".")
import cv2

from config import DetectionConfig
from src.detector import BottleDetector

VAL_IMAGES = "dataset/images/val"
PHASES = {
    "standing": (0, 7),
    "falling": (8, 14),
    "tilt": (15, 17),
    "lying_early": (18, 35),
    "lying_late": (36, 49),
}


def evaluate(model_path: str, conf: float, imgsz: int) -> dict:
    d = BottleDetector(DetectionConfig(model_path=model_path,
                                       conf_threshold=conf, imgsz=imgsz))
    d.load()
    from glob import glob
    hits = {name: 0 for name in PHASES}
    totals = {name: 0 for name in PHASES}
    for phase, (lo, hi) in PHASES.items():
        for i in range(lo, hi + 1):
            path = f"{VAL_IMAGES}/v1_{i:04d}.jpg"
            img = cv2.imread(path)
            if img is None:
                continue
            totals[phase] += 1
            if d.detect(img):
                hits[phase] += 1
    return {p: (hits[p], totals[p]) for p in PHASES}


def main() -> None:
    conf, imgsz = 0.25, 640
    print(f"conf={conf} imgsz={imgsz}")
    print(f"{'phase':12s} {'pretrained':>14s} {'fine-tuned':>14s}")
    pre = evaluate("yolov8n.pt", conf, imgsz)
    ft = evaluate("runs_ft/bottle_ft/weights/best.pt", conf, imgsz)
    for phase in PHASES:
        p = f"{pre[phase][0]}/{pre[phase][1]}"
        f = f"{ft[phase][0]}/{ft[phase][1]}"
        print(f"{phase:12s} {p:>14s} {f:>14s}")


if __name__ == "__main__":
    main()
