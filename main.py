import sys
import os
from pathlib import Path

from PySide6.QtWidgets import QApplication
from dotenv import load_dotenv

# Always load the .env that lives next to main.py, regardless of the shell's
# current working directory or how the app was launched.
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env", override=False)

from ui.overlay_window import OverlayWindow
from utils.mouse_passthrough import MousePassthroughController
from utils.screen_capture_controls import ScreenCaptureControls


def main():
    # Configure High DPI scaling behavior
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    app = QApplication(sys.argv)
    app.setApplicationName("quntumnintent")
    app.setOrganizationName("CopilotAI")

    # Small startup diagnostics: report presence only, never print secrets.
    print(f"[env] .env path: {PROJECT_DIR / '.env'}")
    print(f"[env] NVIDIA_API_KEY loaded: {bool(os.environ.get('NVIDIA_API_KEY', '').strip())}")
    print(f"[env] GEMINI_API_KEY loaded: {bool(os.environ.get('GEMINI_API_KEY', '').strip())}")

    window = OverlayWindow()
    window.show()

    # Restore the existing manual screen capture button + Ctrl+Shift+S wiring.
    screen_capture_controls = ScreenCaptureControls(window)
    window.screen_capture_controls = screen_capture_controls

    # Windows-only selective click-through controller. Non-interactive glass
    # regions let the underlying application own the cursor and receive clicks;
    # title bar/buttons/inputs/resizing controls remain usable.
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
