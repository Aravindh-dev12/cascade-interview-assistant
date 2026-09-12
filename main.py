import os
import signal
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
from utils.candidate_context import install_candidate_context
from utils.inference_watchdog import install_inference_watchdog
from utils.local_qwen_pipeline import install_local_qwen_pipeline
from utils.mouse_passthrough import MousePassthroughController
from utils.parakeet_runtime import install_parakeet_runtime
from utils.realtime_multimodal import (
    CameraVisionControls,
    ensure_default_system_audio,
    install_settings_device_compat,
)
from utils.runtime_reliability import install_audio_device_recovery
from utils.screen_capture_controls import ScreenCaptureControls
from utils.visual_autopilot import install_visual_autopilot


class TooltipBlocker(QObject):
    """Suppress all Qt hover tooltips so scrollable forms stay visually clean."""

    def eventFilter(self, watched, event):
        try:
            if event.type() == QEvent.Type.ToolTip:
                return True
        except KeyboardInterrupt:
            app = QApplication.instance()
            if app is not None:
                app.quit()
            return True
        return super().eventFilter(watched, event)


def _install_console_signal_handlers(app):
    """Let Ctrl+C / console termination shut down the Qt event loop cleanly."""

    def request_quit(signum, _frame):
        print(f"\n[main] Console signal {signum} received; shutting down...")
        app.quit()

    signal.signal(signal.SIGINT, request_quit)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_quit)

    signal_timer = QTimer(app)
    signal_timer.setInterval(100)
    signal_timer.timeout.connect(lambda: None)
    signal_timer.start()
    app._signal_timer = signal_timer


def main():
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    app = QApplication(sys.argv)
    app.setApplicationName("quntumnintent")
    app.setOrganizationName("CopilotAI")
    _install_console_signal_handlers(app)

    tooltip_blocker = TooltipBlocker(app)
    app.installEventFilter(tooltip_blocker)
    app._tooltip_blocker = tooltip_blocker
    install_settings_device_compat()
    install_audio_device_recovery()
    install_visual_autopilot()
    install_inference_watchdog()
    install_candidate_context()
    install_local_qwen_pipeline()
    install_parakeet_runtime()

    print(f"[env] project dir: {PROJECT_DIR}")
    print(f"[env] env file: {env_status['selected_path'] or 'NOT FOUND'}")
    print(f"[env] env exists: {env_status['exists']}")
    print(f"[env] detected names: {', '.join(env_status['detected_names']) or 'none'}")
    print(f"[env] NVIDIA_API_KEY loaded: {env_status['nvidia_loaded']}")
    print(f"[env] PRACTICE_MODE enabled: {env_status['practice_mode']}")
    print("[env] AI_PROVIDER: ollama (forced final answer engine)")
    print(f"[env] OLLAMA_MODEL: {os.environ.get('OLLAMA_MODEL', 'qwen3.5:4b')}")
    print(
        f"[env] NVIDIA_VISION_MODEL: "
        f"{os.environ.get('NVIDIA_VISION_MODEL', 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning')}"
    )
    print(
        f"[env] NVIDIA_RIVA_FUNCTION_ID: "
        f"{os.environ.get('NVIDIA_RIVA_FUNCTION_ID', '1598d209-5e27-4d3c-8079-4751568b1081')}"
    )
    if not env_status["exists"]:
        print(
            "[env] WARNING: no project .env found. Copy .env.template to .env, "
            "then set PRACTICE_MODE=1 and NVIDIA_API_KEY."
        )

    window = OverlayWindow()
    ensure_default_system_audio(window)
    window.show()

    screen_capture_controls = ScreenCaptureControls(window)
    window.screen_capture_controls = screen_capture_controls

    camera_vision_controls = CameraVisionControls(window)
    window.camera_vision_controls = camera_vision_controls

    mouse_passthrough = MousePassthroughController(window)
    window.mouse_passthrough_controller = mouse_passthrough

    audio_device_monitor = AudioDeviceMonitor(window)
    window.audio_device_monitor = audio_device_monitor

    camera_device_monitor = CameraDeviceMonitor(window)
    window.camera_device_monitor = camera_device_monitor

    runtime_label_timer = QTimer(window)
    runtime_label_timer.setInterval(1200)

    def refresh_runtime_state():
        window._configure_ai()
        window.mode_label.setText(window.copilot_ai.runtime_label())
        camera_vision_controls.sync_from_settings()

    runtime_label_timer.timeout.connect(refresh_runtime_state)
    runtime_label_timer.start()
    window.runtime_label_timer = runtime_label_timer
    refresh_runtime_state()

    window.raise_()
    window.activateWindow()

    print("[main] quntumnintent running in manual-first mode.")
    print("Listen: starts Parakeet microphone + system-audio transcription.")
    print("Stop: stops microphone + system-audio transcription.")
    print("Capture screen / Ctrl+Shift+S: captures one screenshot and answers it immediately.")
    print("Send: answers the typed chat prompt using local Qwen.")
    print("Answers: local Qwen 3.5/Ollama only.")
    print("Vision: NVIDIA Nemotron Omni first; local Qwen vision fallback if NVIDIA is busy.")
    print("No automatic listening and no background screenshot inference.")
    print("Candidate context: data/candidate_context.local.md (git-ignored local file).")
    print("Ctrl+Shift+A: toggle Listen/Stop.")
    print("Camera button: analyze the latest live camera frame.")
    print("Ctrl+C in this console: quit cleanly.")

    try:
        exit_code = app.exec()
    finally:
        camera_vision_controls.stop()
        if window.isVisible():
            window.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
