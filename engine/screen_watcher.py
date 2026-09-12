import io
import threading
import time

from PIL import Image, ImageDraw
from PySide6.QtCore import QThread, Signal

from engine.screen_grabber import capture_screen, monitor_for_point


class ScreenWatcher(QThread):
    """Continuously watch a region/monitor and emit stable, meaningful visual context."""

    frame_ready = Signal(bytes)
    status_updated = Signal(str)
    error_occurred = Signal(str)

    def __init__(
        self,
        interval_ms=650,
        stable_ms=450,
        change_threshold=0.055,
        parent=None,
    ):
        super().__init__(parent)
        self.interval_seconds = max(0.25, int(interval_ms) / 1000.0)
        self.stable_seconds = max(0.2, int(stable_ms) / 1000.0)
        self.change_threshold = max(0.01, min(0.5, float(change_threshold)))
        self.running = False
        self._target_lock = threading.Lock()
        self._region = None
        self._point = None
        self._exclude_rect = None
        self._last_emitted_fp = None
        self._pending_fp = None
        self._pending_since = None
        self._pending_image = None

    def set_capture_target(self, region=None, point=None):
        with self._target_lock:
            self._region = dict(region) if region else None
            self._point = tuple(point) if point else None
        self.reset_baseline()

    def set_exclude_rect(self, rect=None):
        """Set a global (left, top, width, height) rectangle to mask from continuous frames."""
        with self._target_lock:
            self._exclude_rect = tuple(rect) if rect else None
        self.reset_baseline()

    @staticmethod
    def _mask_excluded(image, capture_bounds, exclude_rect):
        if not exclude_rect or not capture_bounds:
            return image
        left, top, width, height = exclude_rect
        cap_left = int(capture_bounds.get("left", 0))
        cap_top = int(capture_bounds.get("top", 0))
        x1 = max(0, int(left) - cap_left)
        y1 = max(0, int(top) - cap_top)
        x2 = min(image.width, x1 + int(width))
        y2 = min(image.height, y1 + int(height))
        if x2 <= x1 or y2 <= y1:
            return image
        masked = image.copy()
        ImageDraw.Draw(masked).rectangle((x1, y1, x2, y2), fill=(16, 16, 16))
        return masked

    def reset_baseline(self):
        self._last_emitted_fp = None
        self._pending_fp = None
        self._pending_since = None
        self._pending_image = None

    def stop(self):
        self.running = False
        if self.isRunning():
            self.wait(1200)

    @staticmethod
    def _fingerprint(image: Image.Image) -> bytes:
        return image.convert("L").resize((32, 32), Image.Resampling.BILINEAR).tobytes()

    @staticmethod
    def _distance(left: bytes, right: bytes) -> float:
        if not left or not right or len(left) != len(right):
            return 1.0
        return sum(abs(a - b) for a, b in zip(left, right)) / (255.0 * len(left))

    @staticmethod
    def _encode(image: Image.Image) -> bytes:
        if image.width > 1440:
            ratio = 1440.0 / image.width
            image = image.resize(
                (1440, max(1, int(image.height * ratio))),
                Image.Resampling.LANCZOS,
            )
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=76, optimize=True)
        return buffer.getvalue()

    def run(self):
        self.running = True
        self.status_updated.emit("Screen watcher active")

        while self.running:
            loop_started = time.monotonic()
            try:
                with self._target_lock:
                    region = dict(self._region) if self._region else None
                    point = tuple(self._point) if self._point else None
                    exclude_rect = tuple(self._exclude_rect) if self._exclude_rect else None

                if region:
                    capture_bounds = dict(region)
                elif point:
                    capture_bounds = monitor_for_point(point[0], point[1])
                else:
                    capture_bounds = None
                image = capture_screen(region=region, point=point)
                image = self._mask_excluded(image, capture_bounds, exclude_rect)
                fp = self._fingerprint(image)
                now = time.monotonic()

                if self._last_emitted_fp is None:
                    # Emit the initial visual context instead of silently using it only
                    # as a baseline. This lets an MCQ/coding question already visible at
                    # startup be analyzed without waiting for an unrelated screen change.
                    payload = self._encode(image)
                    self._last_emitted_fp = fp
                    self.frame_ready.emit(payload)
                else:
                    changed_from_last = self._distance(self._last_emitted_fp, fp)
                    if changed_from_last < self.change_threshold:
                        self._pending_fp = None
                        self._pending_since = None
                        self._pending_image = None
                    elif self._pending_fp is None:
                        self._pending_fp = fp
                        self._pending_since = now
                        self._pending_image = image
                    else:
                        pending_motion = self._distance(self._pending_fp, fp)
                        stability_threshold = max(0.012, self.change_threshold * 0.45)
                        if pending_motion > stability_threshold:
                            self._pending_fp = fp
                            self._pending_since = now
                            self._pending_image = image
                        elif now - self._pending_since >= self.stable_seconds:
                            payload = self._encode(image)
                            self._last_emitted_fp = fp
                            self._pending_fp = None
                            self._pending_since = None
                            self._pending_image = None
                            self.frame_ready.emit(payload)
            except Exception as exc:
                self.error_occurred.emit(f"Screen watcher error: {exc}")
                time.sleep(0.5)

            elapsed = time.monotonic() - loop_started
            time.sleep(max(0.03, self.interval_seconds - elapsed))

        self.status_updated.emit("Screen watcher stopped")
