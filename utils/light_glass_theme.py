import ctypes
import os

from PySide6.QtCore import QTimer


def _enable_windows_acrylic(window):
    """Best-effort Windows acrylic blur for appearance only.

    This changes only visual composition. It does not hide the window from capture,
    monitoring, assessment, or accessibility software.
    """
    if os.name != "nt":
        return False

    class ACCENT_POLICY(ctypes.Structure):
        _fields_ = [
            ("AccentState", ctypes.c_int),
            ("AccentFlags", ctypes.c_int),
            ("GradientColor", ctypes.c_uint),
            ("AnimationId", ctypes.c_int),
        ]

    class WINDOWCOMPOSITIONATTRIBDATA(ctypes.Structure):
        _fields_ = [
            ("Attribute", ctypes.c_int),
            ("Data", ctypes.c_void_p),
            ("SizeOfData", ctypes.c_size_t),
        ]

    try:
        hwnd = int(window.winId())
        # ACCENT_ENABLE_ACRYLICBLURBEHIND = 4. GradientColor is AABBGGRR.
        accent = ACCENT_POLICY(4, 0, 0xB8FFFFFF, 0)
        data = WINDOWCOMPOSITIONATTRIBDATA(
            19,
            ctypes.cast(ctypes.pointer(accent), ctypes.c_void_p),
            ctypes.sizeof(accent),
        )
        result = ctypes.windll.user32.SetWindowCompositionAttribute(
            ctypes.c_void_p(hwnd),
            ctypes.byref(data),
        )
        return bool(result)
    except Exception as exc:
        print(f"[ui] Windows acrylic blur unavailable: {exc}")
        return False


def install_light_glass_theme():
    from ui.overlay_window import OverlayWindow

    if getattr(OverlayWindow, "_light_glass_theme_installed", False):
        return

    def update_ui_stylesheet(window):
        font_size = max(11, int(window.settings.get("font_size", 13)))
        window.setStyleSheet(
            f"""
            QWidget {{
                font-family:'Segoe UI Variable Text','Segoe UI',Arial,sans-serif;
                color:#172033;
                font-size:13px;
            }}
            QFrame#container {{
                background:rgba(248,250,252,168);
                border:1px solid rgba(255,255,255,220);
                border-radius:18px;
            }}
            QWidget#titleBar {{
                background:rgba(255,255,255,154);
                border-bottom:1px solid rgba(255,255,255,190);
                border-top-left-radius:18px;
                border-top-right-radius:18px;
            }}
            QWidget#body {{ background:transparent; }}
            QLabel#brandMark {{
                background:rgba(255,255,255,218);
                color:#111827;
                border:1px solid rgba(148,163,184,150);
                border-radius:9px;
                font-weight:800;
            }}
            QLabel#brandTitle {{ color:#111827; font-size:14px; font-weight:750; }}
            QLabel#brandSubtitle, QLabel#muted {{ color:#667085; font-size:10px; }}
            QLabel#eyebrow {{ color:#475467; font-size:10px; font-weight:800; }}
            QFrame#card, QFrame#composer {{
                background:rgba(255,255,255,126);
                border:1px solid rgba(255,255,255,205);
                border-radius:13px;
            }}
            QTextBrowser#answerDisplay {{
                background:transparent;
                border:none;
                color:#182230;
                padding:13px;
                font-size:{font_size}px;
                selection-background-color:rgba(148,163,184,100);
            }}
            QTextBrowser#transcriptDisplay {{
                background:transparent;
                border:none;
                color:#344054;
                font-size:11px;
                selection-background-color:rgba(148,163,184,100);
            }}
            QLabel#partialTranscript {{
                color:#344054;
                background:rgba(255,255,255,135);
                border:1px solid rgba(255,255,255,190);
                border-radius:8px;
                padding:6px 8px;
                font-size:11px;
            }}
            QLineEdit#promptInput {{
                background:transparent;
                border:none;
                color:#111827;
                padding:7px 3px;
            }}
            QLineEdit#promptInput::placeholder {{ color:#98A2B3; }}
            QPushButton {{
                min-height:32px;
                border-radius:9px;
                padding:0 12px;
                font-weight:650;
            }}
            QPushButton#primaryButton, QPushButton#recordBtn {{
                background:rgba(255,255,255,225);
                color:#111827;
                border:1px solid rgba(148,163,184,145);
            }}
            QPushButton#primaryButton:hover, QPushButton#recordBtn:hover {{
                background:rgba(255,255,255,245);
                border-color:rgba(100,116,139,165);
            }}
            QPushButton#secondaryButton, QPushButton#captureBtn, QPushButton#iconButton {{
                background:rgba(255,255,255,92);
                color:#344054;
                border:1px solid rgba(148,163,184,125);
            }}
            QPushButton#secondaryButton:hover, QPushButton#captureBtn:hover, QPushButton#iconButton:hover {{
                background:rgba(255,255,255,190);
                color:#101828;
            }}
            QPushButton#closeButton {{
                background:transparent;
                color:#667085;
                border:1px solid transparent;
                font-size:16px;
            }}
            QPushButton#closeButton:hover {{
                background:rgba(254,226,226,190);
                color:#991B1B;
                border-color:rgba(252,165,165,150);
            }}
            QScrollBar:vertical {{ background:transparent; width:7px; }}
            QScrollBar::handle:vertical {{
                background:rgba(100,116,139,100);
                border-radius:3px;
                min-height:28px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
            """
        )

    def set_status(window, state):
        palette = {
            "IDLE": ("#475467", "rgba(255,255,255,145)", "rgba(148,163,184,120)"),
            "LISTENING": ("#166534", "rgba(240,253,244,190)", "rgba(134,239,172,150)"),
            "THINKING": ("#475467", "rgba(255,255,255,185)", "rgba(148,163,184,130)"),
            "ERROR": ("#991B1B", "rgba(254,242,242,200)", "rgba(252,165,165,150)"),
        }
        fg, bg, border = palette.get(state, palette["IDLE"])
        window.status_badge.setText(state)
        window.status_badge.setStyleSheet(
            f"color:{fg}; background:{bg}; border:1px solid {border}; "
            "border-radius:10px; padding:5px 9px; font-size:10px; font-weight:800;"
        )

    OverlayWindow.update_ui_stylesheet = update_ui_stylesheet
    OverlayWindow._set_status = set_status
    OverlayWindow._light_glass_theme_installed = True


def apply_light_glass(window):
    window.setWindowOpacity(0.98)
    QTimer.singleShot(0, lambda: _enable_windows_acrylic(window))
