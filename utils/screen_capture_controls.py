from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QPushButton, QWidget

from engine.screen_grabber import capture_screen, get_image_bytes


class ScreenCaptureControls:
    """Reliable capture button/hotkey path for the production overlay."""

    def __init__(self, window):
        self.window = window
        self.capture_button = None
        self.capture_in_progress = False
        self._connect_hotkey()
        self._add_capture_button()

    def _connect_hotkey(self):
        try:
            self.window.hotkey_signaler.capture_hotkey_triggered.connect(
                self.capture_and_answer
            )
            print("[capture] Ctrl+Shift+S reliable screen capture connected.")
        except Exception as exc:
            print(f"[capture] Could not connect capture hotkey: {exc}")

    def _add_capture_button(self):
        control_bar = self.window.findChild(QWidget, "controlBar")
        if control_bar is None or control_bar.layout() is None:
            print("[capture] Control bar not found; hotkey remains available.")
            return

        layout = control_bar.layout()
        button = QPushButton("Capture screen")
        button.setObjectName("captureBtn")
        button.clicked.connect(self.capture_and_answer)

        insert_at = min(1, layout.count())
        layout.insertWidget(insert_at, button)
        self.capture_button = button
        print("[capture] Capture Screen button attached.")

    def capture_and_answer(self):
        """Hide the overlay, capture the current work monitor, then restore it."""
        if self.capture_in_progress:
            return
        self.capture_in_progress = True

        if hasattr(self.window, "_configure_gemini"):
            self.window._configure_gemini()
        self.window._set_status("THINKING")

        center = self.window.frameGeometry().center()
        capture_point = (center.x(), center.y())
        saved_region = self.window.settings.get("capture_region")

        # Hide the overlay before MSS grabs pixels. This avoids covering the
        # question and works even if Windows display-affinity protection is not
        # supported by the current GPU/Windows configuration.
        self.window.hide()
        QApplication.processEvents()
        QTimer.singleShot(
            120,
            lambda: self._perform_capture(saved_region, capture_point),
        )

    def _perform_capture(self, saved_region, capture_point):
        try:
            # A saved region remains supported. Without one, capture the monitor
            # where the assistant was located instead of blindly using monitor #1.
            image = capture_screen(
                region=saved_region,
                point=None if saved_region is not None else capture_point,
            )

            # Keep enough resolution for code/MCQ text recognition while avoiding
            # unnecessarily large vision payloads.
            if image.width > 1600:
                from PIL import Image

                ratio = 1600.0 / image.width
                image = image.resize(
                    (1600, max(1, int(image.height * ratio))),
                    Image.Resampling.LANCZOS,
                )

            image_bytes = get_image_bytes(image, format="JPEG", quality=82)
            self.window._enqueue_ai(
                source="Screen capture",
                image_bytes=image_bytes,
            )
        except Exception as exc:
            self.window.answer_display.setMarkdown(
                f"### Screen capture failed\n\n`{exc}`\n\n"
                "Try clearing the saved capture region in Settings and capture again."
            )
            self.window._set_status("ERROR")
        finally:
            self.window.show()
            self.window.raise_()
            self.window.activateWindow()
            if hasattr(self.window, "apply_invisible_mode"):
                self.window.apply_invisible_mode()
            self.capture_in_progress = False
