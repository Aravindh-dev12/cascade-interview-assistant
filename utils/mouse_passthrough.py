import os
import sys

from PySide6.QtCore import QObject


def _experimental_passthrough_enabled() -> bool:
    """Background click-through is opt-in only.

    Earlier builds enabled WS_EX_TRANSPARENT dynamically across large parts of the
    overlay. That caused the mouse cursor from the application behind the overlay
    (for example an I-beam from an editor) to appear inside the assistant and made
    the UI feel like it was moving/typing on the background window.

    Keep the overlay fully interactive by default. The old behavior is intentionally
    disabled even when legacy OVERLAY_MOUSE_PASSTHROUGH=1 exists in an older .env.
    A future explicit implementation can re-introduce a safer per-widget mode.
    """
    return False


class MousePassthroughController(QObject):
    """Compatibility controller that keeps the overlay as a normal UI surface."""

    def __init__(self, window, interval_ms: int = 25):
        super().__init__(window)
        self.window = window
        self.enabled = sys.platform == "win32" and _experimental_passthrough_enabled()
        print("[mouse] Overlay cursor isolation enabled; background passthrough is off.")

    def stop(self):
        pass
