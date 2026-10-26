"""Visual comparison: pretrained vs fine-tuned boxes on sample val frames."""
import sys

sys.path.insert(0, ".")
import cv2

from config import DetectionConfig
from src.detector import BottleDetector

SAMPLES = ["v1_0000", "v1_0010", "v1_0016", "v1_0030", "v1_0045"]


def annotate(model_path: str, conf: float, imgsz: int, color):
    d = BottleDetector(DetectionConfig(model_path=model_path,
                                       conf_threshold=conf, imgsz=imgsz))
    d.load()
    out = []
    for stem in SAMPLES:
        img = cv2.imread(f"dataset/images/val/{stem}.jpg")
        for det in d.detect(img):
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{det.confidence:.2f}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        img = cv2.resize(img, (190, 338))
        out.append(img)
    return out


def main() -> None:
    pre = annotate("yolov8n.pt", 0.25, 640, (0, 0, 255))
    ft = annotate("runs_ft/bottle_ft/weights/best.pt", 0.25, 640, (0, 255, 0))
    import numpy as np
    cv2.imwrite("videos/output/compare_pre_ft.jpg", np.vstack([np.hstack(pre),
                                                               np.hstack(ft)]))
    print("saved videos/output/compare_pre_ft.jpg (red=pretrained, green=fine-tuned)")


if __name__ == "__main__":
    main()
