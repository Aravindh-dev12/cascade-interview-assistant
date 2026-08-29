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
        """Hide the overlay, capture its current monitor, then restore it."""
        if self.capture_in_progress:
            return
        self.capture_in_progress = True

        if hasattr(self.window, "_configure_gemini"):
            self.window._configure_gemini()
        self.window._set_status("THINKING")

        center = self.window.frameGeometry().center()
        capture_point = (center.x(), center.y())

        self.window.hide()
        QApplication.processEvents()
        QTimer.singleShot(120, lambda: self._perform_capture(capture_point))

    def _perform_capture(self, capture_point):
        try:
            # The main Capture Screen action deliberately ignores any previously
            # saved crop. It captures the complete monitor containing the overlay,
            # so a stale region cannot make the app miss the visible problem.
            image = capture_screen(region=None, point=capture_point)

            # Preserve enough pixels for small code/MCQ text while keeping the
            # Gemini request reasonably small and fast.
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
                f"### Screen capture failed\n\n`{exc}`"
            )
            self.window._set_status("ERROR")
        finally:
            self.window.show()
            self.window.raise_()
            self.window.activateWindow()
            if hasattr(self.window, "apply_invisible_mode"):
                self.window.apply_invisible_mode()
            self.capture_in_progress = False
