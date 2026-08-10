from PySide6.QtWidgets import QPushButton, QSizeGrip, QWidget


class ScreenCaptureControls:
    """Restore manual screen-capture entry points for the existing overlay.

    The overlay already owns trigger_screen_analysis(). This helper only
    reconnects the global capture hotkey and adds the missing Capture Screen
    button to the bottom toolbar. The answer engine itself still enforces the
    app's practice/permitted-use mode.
    """

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
            print("[capture] Bottom control bar not found; hotkey remains available.")
            return

        layout = control_bar.layout()
        button = QPushButton("📸 Capture Screen")
        button.setObjectName("captureBtn")
        button.setToolTip("Capture the configured screen region and analyze it with Gemini")
        button.clicked.connect(self.window.trigger_screen_analysis)

        # Put the button after Start Listening and before utility controls.
        insert_at = min(1, layout.count())
        layout.insertWidget(insert_at, button)
        self.capture_button = button
        print("[capture] Capture Screen button restored.")
