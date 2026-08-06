import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from dotenv import load_dotenv

from ui.overlay_window import OverlayWindow
from utils.mouse_passthrough import MousePassthroughController

def main():
    # Load .env file if it exists to fetch default API keys
    load_dotenv()
    
    # Configure High DPI scaling behavior
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    
    # Initialize PySide6 QApplication
    app = QApplication(sys.argv)
    
    # Set application-wide metadata
    app.setApplicationName("quntumnintent")
    app.setOrganizationName("CopilotAI")
    
    # Create the floating overlay window
    window = OverlayWindow()
    window.show()

    # Windows-only selective click-through controller. Non-interactive glass
    # regions let the underlying application own the cursor and receive clicks;
    # title bar/buttons/inputs/resizing controls remain usable.
    mouse_passthrough = MousePassthroughController(window)
    window.mouse_passthrough_controller = mouse_passthrough
    
    # Force the window to stay on top initially
    window.raise_()
    window.activateWindow()
    
    print("[main] quntumnintent running.")
    print("Press Ctrl+Shift+S globally to Capture Region & Answer.")
    print("Press Ctrl+Shift+A globally to Toggle Voice Listening.")
    
    # Run application main event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
