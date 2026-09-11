import os
import time

from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QPushButton, QWidget

import config
from engine.audio_recorder import AudioRecorder, DEFAULT_SPEAKER_LOOPBACK_INDEX
from engine.camera_capture import CameraCaptureController
from engine.copilot_ai import SYSTEM_PROMPT
from engine.nvidia_kimi import NvidiaKimiClient
from engine.visual_context import combine_visual_context


def ensure_default_system_audio(window):
    """Upgrade existing settings to the default-speaker loopback when system audio is disabled."""
    current_system = int(window.settings.get("system_device_idx", -1))
    if current_system != -1:
        return
    try:
        _, detected_system = AudioRecorder.auto_detect_devices()
    except Exception as exc:
        print(f"[audio] Could not auto-select system audio: {exc}")
        return
    if int(detected_system) == -1:
        return
    window.settings["system_device_idx"] = int(detected_system)
    window.audio_recorder.set_devices(
        window.settings.get("mic_device_idx", -1),
        int(detected_system),
    )
    config.save_settings(window.settings)
    print(f"[audio] System audio auto-selected: {detected_system}")


def install_settings_device_compat():
    """Expose the synthetic -2 default-speaker loopback in the older Settings UI."""
    from ui.settings_dialog import SettingsDialog

    if getattr(SettingsDialog, "_default_loopback_compat_installed", False):
        return

    original_load_devices = SettingsDialog._load_devices

    def load_devices(dialog):
        original_load_devices(dialog)
        combo = getattr(dialog, "system_combo", None)
        if combo is None or combo.findData(DEFAULT_SPEAKER_LOOPBACK_INDEX) >= 0:
            return
        try:
            _, loopbacks = AudioRecorder.list_devices()
            item = next(
                (
                    source
                    for source in loopbacks
                    if int(source.get("index", -1))
                    == DEFAULT_SPEAKER_LOOPBACK_INDEX
                ),
                None,
            )
            if item is None:
                return
            combo.insertItem(
                1,
                f"{item['name']} · {item.get('api', 'WASAPI loopback')}",
                DEFAULT_SPEAKER_LOOPBACK_INDEX,
            )
            selected = int(dialog.settings.get("system_device_idx", -1))
            selected_index = combo.findData(selected)
            if selected_index >= 0:
                combo.setCurrentIndex(selected_index)
        except Exception as exc:
            print(f"[settings] Could not expose default speaker loopback: {exc}")

    SettingsDialog._load_devices = load_devices
    SettingsDialog._default_loopback_compat_installed = True


def install_local_provider_compat(window):
    """Install Kimi K3 + local-Qwen routing behind the existing overlay queue.

    Provider modes:
      hybrid -> NVIDIA Kimi K3 first, local Ollama/Qwen fallback
      nvidia -> NVIDIA Kimi K3 only
      ollama -> local Ollama only
      gemini -> Gemini only
      auto   -> hybrid when an NVIDIA key exists, otherwise existing local-first logic

    The older overlay checks ``get_effective_gemini_key`` as a generic AI-ready
    gate. A sentinel keeps that legacy check satisfied for non-Gemini providers;
    no sentinel is ever sent to NVIDIA or Gemini.
    """

    if getattr(window, "_hybrid_provider_installed", False):
        return

    original_key_getter = window.get_effective_gemini_key
    original_stream = window.copilot_ai._local_or_fallback_stream
    original_runtime_label = window.copilot_ai.runtime_label
    kimi = NvidiaKimiClient()
    window.nvidia_kimi = kimi

    def actual_gemini_key():
        return (
            os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        )

    def nvidia_key():
        return os.environ.get("NVIDIA_API_KEY", "").strip()

    def requested_provider():
        env_provider = os.environ.get("AI_PROVIDER", "").strip().lower()
        provider = env_provider or str(
            window.settings.get("ai_provider", "hybrid") or "hybrid"
        ).strip().lower()
        if provider not in {"auto", "hybrid", "nvidia", "ollama", "gemini"}:
            provider = "hybrid"
        if provider == "auto" and nvidia_key():
            return "hybrid"
        return provider

    def compatible_key():
        provider = requested_provider()
        key = actual_gemini_key()
        if provider in {"hybrid", "nvidia", "ollama"}:
            return key or "__AI_PROVIDER_READY__"
        return key or original_key_getter()

    def qwen_stream(prompt, max_tokens, image_bytes_list=None):
        ready, detail = window.copilot_ai.ollama.available()
        if not ready:
            raise RuntimeError(
                f"Local Qwen fallback unavailable: {detail}. "
                f"Run `ollama pull {window.copilot_ai.ollama.model}` and start Ollama."
            )
        yield from window.copilot_ai.ollama.chat_stream(
            prompt,
            max_tokens=max_tokens,
            image_bytes_list=image_bytes_list,
        )

    def routed_stream(prompt, max_tokens, image_bytes_list=None):
        provider = requested_provider()
        if provider not in {"hybrid", "nvidia"}:
            yield from original_stream(
                prompt,
                max_tokens=max_tokens,
                image_bytes_list=image_bytes_list,
            )
            return

        kimi.reconfigure(api_key=nvidia_key())
        if kimi.available():
            yielded = False
            try:
                window.copilot_ai._active_provider = "nvidia"
                for piece in kimi.chat_stream(
                    prompt,
                    max_tokens=max_tokens,
                    image_bytes_list=image_bytes_list,
                    system_prompt=SYSTEM_PROMPT,
                ):
                    yielded = True
                    yield piece
                if yielded:
                    return
            except Exception as exc:
                if provider == "nvidia":
                    raise RuntimeError(f"NVIDIA Kimi K3 request failed: {exc}") from exc
                if yielded:
                    print(f"[nvidia-kimi] Stream interrupted after output began: {exc}")
                    return
                print(f"[nvidia-kimi] Fast cloud path unavailable; falling back to Qwen: {exc}")
        elif provider == "nvidia":
            raise RuntimeError(
                "NVIDIA_API_KEY is missing. Rotate the exposed key and put the new key in the project .env file."
            )

        window.copilot_ai._active_provider = "ollama"
        try:
            yield from qwen_stream(
                prompt,
                max_tokens=max_tokens,
                image_bytes_list=image_bytes_list,
            )
        except Exception as exc:
            raise RuntimeError(
                "Hybrid AI failed: Kimi K3 was unavailable and local Qwen fallback failed: "
                f"{exc}"
            ) from exc

    def configure_ai():
        provider = requested_provider()
        kimi.reconfigure(api_key=nvidia_key())
        underlying_provider = "ollama" if provider in {"hybrid", "nvidia"} else provider
        window.copilot_ai.set_config(
            provider=underlying_provider,
            model=window.settings.get("model", config.DEFAULT_GEMINI_MODEL),
            api_key=actual_gemini_key(),
        )
        window.copilot_ai._requested_provider = provider

    def runtime_label():
        provider = requested_provider()
        if provider == "nvidia":
            return f"NVIDIA · {kimi.model}"
        if provider == "hybrid":
            ready, _, _ = window.copilot_ai.ollama.cached_available()
            fallback = window.copilot_ai.ollama.model if ready else "Qwen fallback"
            return f"HYBRID · Kimi K3 → {fallback}"
        return original_runtime_label()

    window.get_effective_gemini_key = compatible_key
    window._configure_gemini = configure_ai
    window.copilot_ai._local_or_fallback_stream = routed_stream
    window.copilot_ai.runtime_label = runtime_label
    window._hybrid_provider_installed = True
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
