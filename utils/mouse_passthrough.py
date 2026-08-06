import ctypes
import os
import sys
from ctypes import wintypes

from PySide6.QtCore import QObject, QPoint, QTimer
from PySide6.QtWidgets import (
    QAbstractButton,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QScrollBar,
    QSizeGrip,
    QSlider,
    QSpinBox,
)


GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020

SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020


class _POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


def _env_enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class MousePassthroughController(QObject):
    """Selective click-through behavior for the Windows glass overlay.

    The overlay remains visible, but its non-interactive glass/content area is
    temporarily marked WS_EX_TRANSPARENT. Windows then hit-tests the application
    underneath, so Notepad can show its I-beam cursor, browsers can show link
    cursors, and clicks/wheel events reach the background application.

    Interactive overlay controls remain normal Qt widgets: title bar, buttons,
    text input, sliders, scrollbars, combo boxes, checkboxes, and resize grips.
    """

    def __init__(self, window, interval_ms: int = 25):
        super().__init__(window)
        self.window = window
        self.enabled = sys.platform == "win32" and _env_enabled(
            "OVERLAY_MOUSE_PASSTHROUGH", "1"
        )
        self._transparent = None

        self.timer = QTimer(self)
        self.timer.setInterval(max(15, interval_ms))
        self.timer.timeout.connect(self._poll_cursor)

        if self.enabled:
            self.timer.start()
            print("[mouse] Selective background cursor/click passthrough enabled.")
        else:
            print("[mouse] Selective passthrough disabled.")

    def stop(self):
        self.timer.stop()
        self._set_window_transparent(False)

    def _get_cursor_position(self):
        point = _POINT()
        if not ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
            return None
        return point.x, point.y

    def _is_interactive_widget(self, widget) -> bool:
        interactive_types = (
            QAbstractButton,
            QCheckBox,
            QComboBox,
            QLineEdit,
            QScrollBar,
            QSizeGrip,
            QSlider,
            QSpinBox,
        )

        current = widget
        while current is not None and current is not self.window:
            if isinstance(current, interactive_types):
                return True

            # The entire title bar stays interactive so the frameless window can
            # still be dragged even when the content area is click-through.
            if current.objectName() == "titleBar":
                return True

            # Future widgets can opt in without changing this controller.
            if bool(current.property("overlayInteractive")):
                return True

            current = current.parentWidget()

        return False

    def _poll_cursor(self):
        if not self.enabled:
            return

        if not self.window.isVisible() or self.window.isMinimized():
            self._set_window_transparent(False)
            return

        pos = self._get_cursor_position()
        if pos is None:
            return

        global_point = QPoint(pos[0], pos[1])
        local_point = self.window.mapFromGlobal(global_point)

        if not self.window.rect().contains(local_point):
            # Keep the overlay interactive while the pointer is outside it. The
            # timer can still switch it to passthrough immediately on re-entry.
            self._set_window_transparent(False)
            return

        child = self.window.childAt(local_point)
        should_passthrough = not self._is_interactive_widget(child)
        self._set_window_transparent(should_passthrough)

    def _set_window_transparent(self, enabled: bool):
        if sys.platform != "win32" or self._transparent == enabled:
            return

        hwnd = int(self.window.winId())
        user32 = ctypes.windll.user32

        # Use pointer-sized APIs on 64-bit Python and fall back on 32-bit APIs.
        get_style = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
        set_style = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)

        current_style = get_style(hwnd, GWL_EXSTYLE)
        if enabled:
            new_style = current_style | WS_EX_TRANSPARENT
        else:
            new_style = current_style & ~WS_EX_TRANSPARENT

        if new_style != current_style:
            set_style(hwnd, GWL_EXSTYLE, new_style)
            # Apply the changed extended style without moving, resizing, raising,
            # activating, or recreating the Qt window.
            user32.SetWindowPos(
                hwnd,
                0,
                0,
                0,
                0,
                0,
                SWP_NOMOVE
                | SWP_NOSIZE
                | SWP_NOZORDER
                | SWP_NOACTIVATE
                | SWP_FRAMECHANGED,
            )

        self._transparent = enabled
