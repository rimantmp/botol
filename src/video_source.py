"""Video input: webcam and recorded file, with controlled reconnect for webcam."""
import time
from typing import Optional

import cv2


class VideoReadError(Exception):
    """Raised when a video source cannot be opened or read."""


class VideoSource:
    """Uniform frame iterator over webcam or file.

    - Webcam: reconnects up to `max_reconnects` times with a delay.
    - File: verifies existence, stops cleanly at end of file.
    Reading logic is kept separate from detection logic.
    """

    def __init__(self, source, reconnect_delay: float = 2.0, max_reconnects: int = 5):
        self.source = source
        self.is_webcam = isinstance(source, int)
        self.reconnect_delay = reconnect_delay
        self.max_reconnects = max_reconnects
        self.cap: Optional[cv2.VideoCapture] = None
        self._reconnects_used = 0

    def open(self) -> None:
        if not self.is_webcam and not self._file_exists():
            raise VideoReadError(f"Video file not found: {self.source}")
        if not self._open_capture():
            raise VideoReadError(f"Cannot open video source: {self.source!r}")

    def _file_exists(self) -> bool:
        import os
        return os.path.isfile(self.source)

    def _open_capture(self) -> bool:
        # On Windows prefer CAP_DSHOW for webcams (faster open, no MSMF delay).
        if self.is_webcam:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.source)
        return self.cap.isOpened()

    def read(self) -> tuple[bool, Optional["cv2.MatLike"]]:
        """Read one frame. Returns (ok, frame). Handles EOF and webcam drops."""
        ok, frame = self.cap.read()
        if ok:
            return True, frame

        if not self.is_webcam:
            return False, None  # file ended

        # Webcam dropped: try controlled reconnect.
        while self._reconnects_used < self.max_reconnects:
            self._reconnects_used += 1
            time.sleep(self.reconnect_delay)
            self.release()
            if self._open_capture():
                ok, frame = self.cap.read()
                if ok:
                    return True, frame
        return False, None

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False
