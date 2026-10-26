"""Generate YOLO labels for the bottle fine-tune dataset.

Boxes were measured manually from grid overlays on key frames of the two
source videos (see videos/output/measure_*.jpg) and interpolated across
phases: standing -> falling (motion blur) -> lying.

YOLO format per image: one .txt with lines "0 cx cy w h" (normalized).
Class 0 = bottle (single-class dataset).
"""
import os
from glob import glob

IMAGES_DIR = "dataset/images"
LABELS_DIR = "dataset/labels"

# Measured normalized boxes (x1, y1, x2, y2) per phase.
# v2 revision: blur-phase boxes tightened to the actual streak extent so the
# model does not learn "large blurry wall region" as a bottle.
V1_STAND = (0.3616, 0.0000, 0.4860, 0.2390)   # frames 0-7 (cut at top edge)
V1_FALL = (0.44, 0.35, 0.52, 0.90)            # frames 8-14 (vertical blur streak)
V1_TILT = (0.33, 0.685, 0.63, 0.91)           # frames 15-17 (diagonal on bed)
V1_LYING_EARLY = (0.15, 0.965, 0.55, 1.00)    # frames 18-35 (sliver at bottom)
V1_LYING_LATE = (0.08, 0.965, 0.45, 1.00)     # frames 36-49 (settled lower-left)

V2_STAND = (0.3566, 0.0000, 0.4958, 0.2291)   # frames 0-8 (cut at top edge)
V2_FALL_HIGH = (0.40, 0.05, 0.50, 0.55)       # frames 9-20 (narrow blur streak)
V2_FALL_LOW = (0.38, 0.45, 0.52, 0.95)        # frames 21-29 (approaching bed)
V2_MIDFALL = (0.42, 0.595, 0.53, 0.9775)      # frame 30 (vertical, above bed)
V2_LYING_EARLY = (0.48, 0.8425, 0.92, 0.955)  # frames 36-45 (landing zone)
V2_LYING_LATE = (0.52, 0.865, 0.98, 0.978)    # frames 46-71 (settled)


def phase_box(tag: str, i: int):
    """Return the measured box for one frame, or None for no-bottle frames."""
    if tag == "v1":
        if i <= 7:
            return V1_STAND
        if i <= 14:
            return V1_FALL
        if i <= 17:
            return V1_TILT
        if i <= 35:
            return V1_LYING_EARLY
        return V1_LYING_LATE
    else:  # v2
        if i <= 8:
            return V2_STAND
        if i <= 20:
            return V2_FALL_HIGH
        if i <= 29:
            return V2_FALL_LOW
        if i == 30:
            return V2_MIDFALL
        if i <= 45:
            return V2_LYING_EARLY
        return V2_LYING_LATE


def to_yolo_line(box) -> str:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    return f"0 {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}"


def main() -> None:
    os.makedirs(LABELS_DIR, exist_ok=True)
    n = 0
    for img_path in sorted(glob(os.path.join(IMAGES_DIR, "*.jpg"))):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        tag, idx = stem[:2], int(stem[3:])
        box = phase_box(tag, idx)
        label_path = os.path.join(LABELS_DIR, stem + ".txt")
        with open(label_path, "w", encoding="utf-8") as f:
            if box is not None:
                f.write(to_yolo_line(box) + "\n")
        n += 1
    print(f"labels written: {n}")


if __name__ == "__main__":
    main()
