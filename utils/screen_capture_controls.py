import ctypes
import os
from ctypes import wintypes

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QPushButton, QWidget

from engine.screen_grabber import capture_screen, get_image_bytes, monitor_for_point


class ScreenCaptureControls:
    """Manual screen-capture button/hotkey path.

    On Windows, remember the most recent foreground application that is not the
    overlay itself. This lets a Capture click inside the overlay still target the
    monitor where the user's practice browser/editor was active immediately before
    the click. A configured capture_region always wins.
    """

    def __init__(self, window):
        self.window = window
        self.capture_button = None
        self.capture_in_progress = False
        self.last_external_point = None
        self.last_external_hwnd = None
        self._foreground_timer = None
        self._connect_hotkey()
        self._add_capture_button()
        self._start_foreground_tracking()

    def _connect_hotkey(self):
        try:
            self.window.hotkey_signaler.capture_hotkey_triggered.connect(self.capture_and_answer)
        except Exception as exc:
            print(f"[capture] Could not connect capture hotkey: {exc}")

    def _add_capture_button(self):
        control_bar = self.window.findChild(QWidget, "controlBar")
        if control_bar is None or control_bar.layout() is None:
            return
        button = QPushButton("Capture screen")
        button.setObjectName("captureBtn")
        button.clicked.connect(self.capture_and_answer)
        control_bar.layout().insertWidget(min(1, control_bar.layout().count()), button)
        self.capture_button = button

    def _start_foreground_tracking(self):
        if os.name != "nt":
            return
        try:
            timer = QTimer(self.window)
            timer.setInterval(200)
            timer.timeout.connect(self._remember_external_foreground)
            timer.start()
            self._foreground_timer = timer
            self._remember_external_foreground()
            print("[capture] Foreground-window tracking enabled for manual screen capture.")
        except Exception as exc:
            print(f"[capture] Foreground-window tracking unavailable: {exc}")

    def _remember_external_foreground(self):
        if os.name != "nt":
            return
        try:
            user32 = ctypes.windll.user32
            hwnd = int(user32.GetForegroundWindow() or 0)
            if not hwnd:
                return

            overlay_hwnd = int(self.window.winId())
            if hwnd == overlay_hwnd:
                return

            rect = wintypes.RECT()
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return
            width = int(rect.right - rect.left)
            height = int(rect.bottom - rect.top)
            if width < 100 or height < 100:
                return

            self.last_external_hwnd = hwnd
            self.last_external_point = (
                int(rect.left + width // 2),
                int(rect.top + height // 2),
            )
        except Exception:
            # Tracking is best-effort; capture always has a deterministic fallback.
            return

    def _capture_point(self):
        if self.last_external_point is not None:
            return self.last_external_point, "last foreground app"
        center = self.window.frameGeometry().center()
        return (center.x(), center.y()), "overlay monitor fallback"

    def capture_and_answer(self):
        if self.capture_in_progress:
            return
        self.capture_in_progress = True
        region = self.window.settings.get("capture_region")
        point, target_source = self._capture_point()

        # Keep the overlay visible. Capture protection/masking is handled by the
        # normal window/capture pipeline; manual capture should not make the UI vanish.
        self._perform_capture(region, point, target_source)

    def _perform_capture(self, region, point, target_source):
        try:
            if region:
                print(
                    "[capture] Targeting configured capture region · "
                    f"left={region.get('left')} top={region.get('top')} "
                    f"width={region.get('width')} height={region.get('height')}"
                )
            else:
                monitor = monitor_for_point(*point)
                print(
                    f"[capture] Targeting {target_source} · point={point} · "
                    f"monitor=({monitor.get('left')},{monitor.get('top')},"
                    f"{monitor.get('width')}x{monitor.get('height')})"
                )

            image = capture_screen(region=region, point=None if region else point)
            original_size = (image.width, image.height)
            if image.width > 1600:
                from PIL import Image

                ratio = 1600.0 / image.width
                image = image.resize(
                    (1600, max(1, int(image.height * ratio))),
                    Image.Resampling.LANCZOS,
                )
            image_bytes = get_image_bytes(image, format="JPEG", quality=82)
            print(
                f"[capture] Manual screenshot captured · source={original_size[0]}x{original_size[1]} · "
                f"sent={image.width}x{image.height} · bytes={len(image_bytes)}"
            )
            self.window.submit_screen_capture(image_bytes, source="Manual screen capture")
        except Exception as exc:
            self.window.answer_display.setMarkdown(f"### Screen capture failed\n\n`{exc}`")
            self.window._set_status("ERROR")
        finally:
            self.capture_in_progress = False
