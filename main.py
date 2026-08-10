import sys
import os
from pathlib import Path

from PySide6.QtWidgets import QApplication

PROJECT_DIR = Path(__file__).resolve().parent

from utils.env_loader import load_project_env

env_status = load_project_env(PROJECT_DIR)

from ui.overlay_window import OverlayWindow
from utils.mouse_passthrough import MousePassthroughController
from utils.screen_capture_controls import ScreenCaptureControls


def main():
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    app = QApplication(sys.argv)
    app.setApplicationName("quntumnintent")
    app.setOrganizationName("CopilotAI")

    print(f"[env] project dir: {PROJECT_DIR}")
    print(f"[env] env file: {env_status['selected_path'] or 'NOT FOUND'}")
    print(f"[env] env exists: {env_status['exists']}")
    print(f"[env] detected names: {', '.join(env_status['detected_names']) or 'none'}")
    print(f"[env] NVIDIA_API_KEY loaded: {env_status['nvidia_loaded']}")
    print(f"[env] GEMINI_API_KEY loaded: {env_status['gemini_loaded']}")
    print(f"[env] OPENAI_API_KEY loaded: {env_status['openai_loaded']}")
    print(f"[env] PRACTICE_MODE enabled: {env_status['practice_mode']}")

    window = OverlayWindow()
    window.show()

    screen_capture_controls = ScreenCaptureControls(window)
    window.screen_capture_controls = screen_capture_controls

    mouse_passthrough = MousePassthroughController(window)
    window.mouse_passthrough_controller = mouse_passthrough

    window.raise_()
    window.activateWindow()

    print("[main] quntumnintent running.")
    print("Press Ctrl+Shift+S globally to Capture Region & Answer.")
    print("Press Ctrl+Shift+A globally to Toggle Voice Listening.")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
