"""Split dataset by source video: train=v2, val=v1 (held-out drop event)."""
import os
import shutil
from glob import glob

IMAGES = "dataset/images"
LABELS = "dataset/labels"


def link_or_copy(src: str, dst_dir: str) -> None:
    name = os.path.basename(src)
    dst = os.path.join(dst_dir, name)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)


def main() -> None:
    for split, tag in (("train", "v2"), ("val", "v1")):
        img_dir = os.path.join(IMAGES, split)
        lbl_dir = os.path.join(LABELS, split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        n = 0
        for img in sorted(glob(os.path.join(IMAGES, f"{tag}_*.jpg"))):
            link_or_copy(img, img_dir)
            stem = os.path.splitext(os.path.basename(img))[0]
            link_or_copy(os.path.join(LABELS, stem + ".txt"), lbl_dir)
            n += 1
        print(f"{split}: {n} images")


if __name__ == "__main__":
    main()
