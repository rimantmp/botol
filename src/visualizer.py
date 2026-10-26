"""On-screen visualization: boxes, track IDs, status labels, FPS, DB state."""
import cv2

# Consistent status colors (BGR)
STATUS_COLORS = {
    "BERDIRI": (0, 200, 0),          # green
    "JATUH": (0, 0, 255),            # red
    "TIDAK_PASTI": (0, 215, 255),    # yellow
    "TIDAK_TERDETEKSI": (160, 160, 160),  # gray
}


class Visualizer:
    """Draws detections and HUD onto frames."""

    def __init__(self, source_label: str = "unknown", virtual_line_y: float = None):
        self.source_label = source_label
        self.virtual_line_y = virtual_line_y
        self.db_status = "unknown"     # updated by main loop
        self.fps = 0.0

    def draw(self, frame, tracks, statuses: dict, fps: float):
        """Annotate frame in place and return it."""
        self.fps = fps
        self._last_count = len(tracks)

        # Draw virtual drop line if configured
        if self.virtual_line_y is not None:
            ly = int(self.virtual_line_y)
            cv2.line(frame, (0, ly), (frame.shape[1], ly), (255, 255, 0), 2)
            cv2.putText(frame, "BATAS KETINGGIAN (DROP LINE)", (10, max(20, ly - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        for tr in tracks:
            x1, y1, x2, y2 = (int(v) for v in tr.bbox)
            status = statuses.get(tr.track_id, "TIDAK_PASTI")
            color = STATUS_COLORS.get(status, STATUS_COLORS["TIDAK_TERDETEKSI"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw trajectory trail
            if tr.history and len(tr.history) > 1:
                pts = [(int(entry[1]), int(entry[2])) for entry in tr.history[-10:]]
                for p_idx in range(1, len(pts)):
                    cv2.line(frame, pts[p_idx - 1], pts[p_idx], color, 2)

            label = f"ID {tr.track_id} {status}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            y_text = max(y1 - 6, th + 4)
            cv2.rectangle(frame, (x1, y_text - th - 4), (x1 + tw + 4, y_text + 2), color, -1)
            cv2.putText(frame, label, (x1 + 2, y_text - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        self._draw_hud(frame)
        return frame

    def _draw_hud(self, frame) -> None:
        lines = (
            f"Src: {self.source_label}  FPS: {self.fps:.1f}",
            f"Objects: {getattr(self, '_last_count', 0)}",
            f"DB: {self.db_status}",
        )
        for n, line in enumerate(lines):
            cv2.putText(frame, line, (10, 25 + 25 * n),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
