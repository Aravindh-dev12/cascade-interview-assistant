import os
import sys
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtWidgets import QApplication

PROJECT_DIR = Path(__file__).resolve().parent

from utils.env_loader import load_project_env

env_status = load_project_env(PROJECT_DIR)

from ui.overlay_window import OverlayWindow
from utils.audio_device_monitor import AudioDeviceMonitor
from utils.camera_device_monitor import CameraDeviceMonitor
from utils.mouse_passthrough import MousePassthroughController
from utils.screen_capture_controls import ScreenCaptureControls


class TooltipBlocker(QObject):
    """Suppress all Qt hover tooltips so scrollable forms stay visually clean."""

    def eventFilter(self, watched, event):
        if event.type() == QEvent.ToolTip:
            return True
        return super().eventFilter(watched, event)


def main():
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    app = QApplication(sys.argv)
    app.setApplicationName("quntumnintent")
    app.setOrganizationName("CopilotAI")

    tooltip_blocker = TooltipBlocker(app)
    app.installEventFilter(tooltip_blocker)
    app._tooltip_blocker = tooltip_blocker

    print(f"[env] project dir: {PROJECT_DIR}")
    print(f"[env] env file: {env_status['selected_path'] or 'NOT FOUND'}")
    print(f"[env] env exists: {env_status['exists']}")
    print(f"[env] detected names: {', '.join(env_status['detected_names']) or 'none'}")
    print(f"[env] NVIDIA_API_KEY loaded: {env_status['nvidia_loaded']}")
    print(f"[env] GEMINI_API_KEY loaded: {env_status['gemini_loaded']}")
    print(f"[env] PRACTICE_MODE enabled: {env_status['practice_mode']}")

    window = OverlayWindow()
    window.show()

    screen_capture_controls = ScreenCaptureControls(window)
    window.screen_capture_controls = screen_capture_controls

    mouse_passthrough = MousePassthroughController(window)
    window.mouse_passthrough_controller = mouse_passthrough

    audio_device_monitor = AudioDeviceMonitor(window)
    window.audio_device_monitor = audio_device_monitor

    camera_device_monitor = CameraDeviceMonitor(window)
    window.camera_device_monitor = camera_device_monitor

    # Keep the visible model badge truthful even when AUTO switches between local
    # Ollama and Gemini fallback. This avoids touching the request/streaming path.
    runtime_label_timer = QTimer(window)
    runtime_label_timer.setInterval(1500)
    runtime_label_timer.timeout.connect(
        lambda: window.mode_label.setText(window.copilot_ai.runtime_label())
    )
    runtime_label_timer.start()
    window.runtime_label_timer = runtime_label_timer
    window.mode_label.setText(window.copilot_ai.runtime_label())

    window.raise_()
    window.activateWindow()

    if (
        window.settings.get("auto_start_listening", True)
        and env_status["practice_mode"]
        and env_status["nvidia_loaded"]
    ):
        QTimer.singleShot(350, window.toggle_recording)

    print("[main] quntumnintent running.")
    print("Press Ctrl+Shift+S globally to Capture Region & Answer.")
    print("Press Ctrl+Shift+A globally to Toggle Voice Listening.")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
