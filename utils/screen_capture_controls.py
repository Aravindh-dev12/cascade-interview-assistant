from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QPushButton, QWidget

from engine.screen_grabber import capture_screen, get_image_bytes


class ScreenCaptureControls:
    """Manual screen-capture button/hotkey path."""

    def __init__(self, window):
        self.window = window
        self.capture_button = None
        self.capture_in_progress = False
        self._connect_hotkey()
        self._add_capture_button()

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

    def capture_and_answer(self):
        if self.capture_in_progress:
            return
        self.capture_in_progress = True
        center = self.window.frameGeometry().center()
        point = (center.x(), center.y())
        region = self.window.settings.get("capture_region")

        # Hide only for explicit manual captures so the frame is clean even when
        # Windows display-affinity protection is disabled or unsupported.
        self.window.hide()
        QApplication.processEvents()
        QTimer.singleShot(100, lambda: self._perform_capture(region, point))

    def _perform_capture(self, region, point):
        try:
            image = capture_screen(region=region, point=None if region else point)
            if image.width > 1600:
                from PIL import Image

                ratio = 1600.0 / image.width
                image = image.resize(
                    (1600, max(1, int(image.height * ratio))),
                    Image.Resampling.LANCZOS,
                )
            image_bytes = get_image_bytes(image, format="JPEG", quality=82)
            self.window.submit_screen_capture(image_bytes, source="Manual screen capture")
        except Exception as exc:
            self.window.answer_display.setMarkdown(f"### Screen capture failed\n\n`{exc}`")
            self.window._set_status("ERROR")
        finally:
            self.window.show()
            self.window.raise_()
            self.window.activateWindow()
            if hasattr(self.window, "apply_invisible_mode"):
                self.window.apply_invisible_mode()
            self.capture_in_progress = False
