import os
import time

from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QPushButton, QWidget

import config
from engine.camera_capture import CameraCaptureController
from engine.visual_context import combine_visual_context


def install_local_provider_compat(window):
    """Make the existing overlay truly work in Ollama-only mode.

    Older overlay code uses GEMINI_API_KEY presence as a generic AI-ready gate.
    For a local Qwen runtime that gate is unnecessary. This bridge keeps Gemini
    validation for cloud-only mode, while giving the existing queue a harmless
    sentinel key only when the effective runtime is local Ollama.
    """

    original_key_getter = window.get_effective_gemini_key

    def actual_gemini_key():
        return (
            os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        )

    def effective_provider():
        provider = str(window.settings.get("ai_provider", "auto") or "auto").lower()
        if provider == "auto" and not actual_gemini_key():
            return "ollama"
        return provider if provider in {"auto", "ollama", "gemini"} else "auto"

    def compatible_key():
        key = actual_gemini_key()
        if key:
            return key
        if effective_provider() == "ollama":
            return "__LOCAL_OLLAMA__"
        return original_key_getter()

    def configure_ai():
        provider = effective_provider()
        key = actual_gemini_key()
        window.copilot_ai.set_config(
            provider=provider,
            model=window.settings.get("model", config.DEFAULT_GEMINI_MODEL),
            api_key=key or ("__LOCAL_OLLAMA__" if provider == "ollama" else ""),
        )

    window.get_effective_gemini_key = compatible_key
    window._configure_gemini = configure_ai
    configure_ai()


class CameraVisionControls(QObject):
    """Live camera frame capture plus speech/chat visual-context integration."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.latest_camera_bytes = None
        self.latest_camera_time = 0.0
        self.latest_screen_bytes = None
        self.latest_screen_time = 0.0
        self.camera_button = None

        self.camera_capture = CameraCaptureController(
            device_id=window.settings.get("camera_device_id", ""),
            interval_ms=window.settings.get("camera_frame_interval_ms", 450),
            parent=window,
        )
        window.camera_capture = self.camera_capture
        window.latest_camera_bytes = None
        window.latest_camera_time = 0.0

        self.camera_capture.frame_ready.connect(self._on_camera_frame)
        self.camera_capture.status_updated.connect(self._on_status)
        self.camera_capture.error_occurred.connect(self._on_status)

        if hasattr(window, "screen_watcher"):
            window.screen_watcher.frame_ready.connect(self._on_screen_frame)

        self._wrap_manual_screen_capture()
        self._add_camera_button()
        self.sync_from_settings()

    def _practice_mode_enabled(self):
        return os.environ.get("PRACTICE_MODE", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }

    def _add_camera_button(self):
        control_bar = self.window.findChild(QWidget, "controlBar")
        if control_bar is None or control_bar.layout() is None:
            return
        button = QPushButton("Camera")
        button.setObjectName("captureBtn")
        button.clicked.connect(self.analyze_camera)
        layout = control_bar.layout()
        layout.insertWidget(min(2, layout.count()), button)
        self.camera_button = button

    def _wrap_manual_screen_capture(self):
        original = self.window.submit_screen_capture

        def submit(image_bytes, source="Manual screen capture"):
            result = original(image_bytes, source=source)
            self.latest_screen_bytes = image_bytes
            self.latest_screen_time = time.monotonic()
            self._publish_combined_context()
            return result

        self.window.submit_screen_capture = submit

    def sync_from_settings(self):
        self.camera_capture.configure(
            device_id=self.window.settings.get("camera_device_id", ""),
            interval_ms=self.window.settings.get("camera_frame_interval_ms", 450),
        )
        enabled = bool(
            self.window.settings.get("camera_capture_enabled", True)
            and self._practice_mode_enabled()
        )
        if enabled:
            self.camera_capture.start()
        else:
            self.camera_capture.stop()
        if self.camera_button is not None:
            self.camera_button.setEnabled(enabled)

    def stop(self):
        self.camera_capture.stop()

    def _on_status(self, message):
        print(f"[camera] {message}")

    def _on_screen_frame(self, image_bytes):
        # Run after OverlayWindow.handle_screen_frame has received the same signal.
        # Defer one event-loop turn so our combined context becomes the final value.
        QTimer.singleShot(0, lambda payload=image_bytes: self._remember_screen(payload))

    def _remember_screen(self, image_bytes):
        self.latest_screen_bytes = image_bytes
        self.latest_screen_time = time.monotonic()
        self._publish_combined_context()

    def _on_camera_frame(self, image_bytes):
        self.latest_camera_bytes = image_bytes
        self.latest_camera_time = time.monotonic()
        self.window.latest_camera_bytes = image_bytes
        self.window.latest_camera_time = self.latest_camera_time
        self._publish_combined_context()

    def _publish_combined_context(self):
        items = []
        if self.latest_screen_bytes:
            items.append(("SCREEN", self.latest_screen_bytes))

        max_camera_age = float(
            self.window.settings.get("camera_context_max_age_seconds", 3.0)
        )
        if (
            self.latest_camera_bytes
            and time.monotonic() - self.latest_camera_time <= max_camera_age
        ):
            items.append(("CAMERA", self.latest_camera_bytes))

        if not items:
            return
        try:
            combined = combine_visual_context(items)
        except Exception as exc:
            print(f"[camera] Could not combine visual context: {exc}")
            combined = items[-1][1]
        if combined:
            self.window.latest_screen_bytes = combined
            self.window.latest_screen_time = time.monotonic()
            if hasattr(self.window, "screen_meta"):
                if self.latest_camera_bytes and self.latest_screen_bytes:
                    self.window.screen_meta.setText("SCREEN + CAMERA LIVE")
                elif self.latest_camera_bytes:
                    self.window.screen_meta.setText("CAMERA LIVE")

    def analyze_camera(self):
        if not self.latest_camera_bytes:
            self.window.answer_display.setMarkdown(
                "### Camera not ready\n\nNo camera frame has arrived yet. "
                "Check the selected camera and Windows camera permission."
            )
            return
        self.window._enqueue_ai(
            source="Camera frame",
            kind="manual_screen",
            image_bytes=self.latest_camera_bytes,
            custom_query=(
                "Analyze the current camera frame. Read any visible text/code and "
                "identify the object, diagram, document, or question the user is showing. "
                "Answer directly and concisely."
            ),
            use_image_history=False,
        )
