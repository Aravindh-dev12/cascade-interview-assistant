from PySide6.QtWidgets import QPushButton, QWidget


class ScreenCaptureControls:
    """Attach the existing screen-capture action to the production control bar."""

    def __init__(self, window):
        self.window = window
        self.capture_button = None
        self._connect_hotkey()
        self._add_capture_button()

    def _connect_hotkey(self):
        try:
            self.window.hotkey_signaler.capture_hotkey_triggered.connect(
                self.window.trigger_screen_analysis
            )
            print("[capture] Ctrl+Shift+S screen capture connected.")
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
        button.clicked.connect(self.window.trigger_screen_analysis)

        insert_at = min(1, layout.count())
        layout.insertWidget(insert_at, button)
        self.capture_button = button
        print("[capture] Capture Screen button attached.")
