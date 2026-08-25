import os
import time

from PySide6.QtCore import QObject, QPoint, QThread, Qt, Signal, Slot
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizeGrip,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from pynput import keyboard

import config
from engine.audio_recorder import AudioRecorder
from engine.copilot_ai import CopilotAI
from engine.screen_grabber import capture_screen, get_image_bytes
from engine.stt_worker import STTWorker
from ui.settings_dialog import SettingsDialog
from utils.win_utils import set_window_invisible_to_capture


class HotkeySignaler(QObject):
    capture_hotkey_triggered = Signal()
    record_hotkey_triggered = Signal()


class AIQueryWorker(QThread):
    answer_ready = Signal(str)

    def __init__(self, copilot_ai, image_bytes=None, custom_query=None):
        super().__init__()
        self.copilot_ai = copilot_ai
        self.image_bytes = image_bytes
        self.custom_query = custom_query

    def run(self):
        answer = self.copilot_ai.generate_answer(
            image_bytes=self.image_bytes,
            custom_query=self.custom_query,
        )
        self.answer_ready.emit(answer)


class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = config.load_settings()

        self.audio_recorder = AudioRecorder()
        self._ensure_audio_defaults()
        self.audio_recorder.set_devices(
            self.settings.get("mic_device_idx", -1),
            self.settings.get("system_device_idx", -1),
        )

        self.stt_worker = STTWorker(
            self.audio_recorder,
            api_key=self.get_effective_nvidia_key(),
        )
        self.copilot_ai = CopilotAI(
            model=self.settings.get("model", config.DEFAULT_GEMINI_MODEL),
            api_key=self.get_effective_gemini_key(),
        )

        self.drag_position = QPoint()
        self.hotkey_signaler = HotkeySignaler()
        self.hotkey_signaler.record_hotkey_triggered.connect(self.toggle_recording)
        self.hotkey_listener = None

        self.answer_history = []
        self.current_trigger_source = None
        self.last_query_time = 0.0
        self.ai_worker = None

        self._apply_window_flags()
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMinimumSize(380, 460)
        self.resize(430, 700)

        self.init_ui()
        self.setup_global_hotkeys()

        self.stt_worker.partial_transcription_ready.connect(self.handle_partial_transcription)
        self.stt_worker.transcription_ready.connect(self.handle_transcription)
        self.stt_worker.status_updated.connect(self.update_status_log)
        self.stt_worker.error_occurred.connect(self.handle_stt_error)
        self.stt_worker.start()

    def _ensure_audio_defaults(self):
        if (
            self.settings.get("mic_device_idx", -1) == -1
            and self.settings.get("system_device_idx", -1) == -1
        ):
            mic_idx, system_idx = AudioRecorder.auto_detect_devices()
            self.settings["mic_device_idx"] = mic_idx
            self.settings["system_device_idx"] = system_idx

    def _apply_window_flags(self):
        flags = Qt.FramelessWindowHint | Qt.Tool
        if self.settings.get("always_on_top", True):
            flags |= Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)

    def get_effective_gemini_key(self):
        settings_key = self.settings.get("api_key", "").strip()
        if settings_key:
            return settings_key
        return (
            os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        )

    def get_effective_nvidia_key(self):
        settings_key = self.settings.get("nvidia_api_key", "").strip()
        if settings_key:
            return settings_key
        return os.environ.get("NVIDIA_API_KEY", "").strip()

    def setup_global_hotkeys(self):
        try:
            if self.hotkey_listener:
                self.hotkey_listener.stop()
            hotkey_map = {
                self.settings.get("hotkey_capture", "<ctrl>+<shift>+s"): (
                    lambda: self.hotkey_signaler.capture_hotkey_triggered.emit()
                ),
                self.settings.get("hotkey_record", "<ctrl>+<shift>+a"): (
                    lambda: self.hotkey_signaler.record_hotkey_triggered.emit()
                ),
            }
            self.hotkey_listener = keyboard.GlobalHotKeys(hotkey_map)
            self.hotkey_listener.start()
        except Exception as exc:
            print(f"[hotkey] Error registering global hotkeys: {exc}")

    def init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)

        self.container = QFrame(self)
        self.container.setObjectName("container")
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        root_layout.addWidget(self.container)

        title_bar = QWidget()
        title_bar.setObjectName("titleBar")
        title_bar.setProperty("overlayInteractive", True)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(12, 4, 8, 4)
        title_layout.addWidget(QLabel("🤖 quntumnintent · Gemini + NVIDIA"))
        title_layout.addStretch()

        settings_btn = QPushButton("⚙")
        settings_btn.setToolTip("Settings")
        settings_btn.clicked.connect(self.open_settings)
        min_btn = QPushButton("—")
        min_btn.clicked.connect(self.showMinimized)
        close_btn = QPushButton("✕")
        close_btn.clicked.connect(self.close)
        for button in (settings_btn, min_btn, close_btn):
            button.setObjectName("windowBtn")
            title_layout.addWidget(button)

        title_bar.mousePressEvent = self.title_bar_mouse_press
        title_bar.mouseMoveEvent = self.title_bar_mouse_move
        container_layout.addWidget(title_bar)

        self.answer_display = QTextBrowser()
        self.answer_display.setOpenExternalLinks(True)
        self.answer_display.setMarkdown(
            "### Ready\n\n"
            "Gemini handles text and screen reasoning. NVIDIA Nemotron handles "
            "real-time speech transcription. Configure both API keys in Settings or `.env`."
        )
        container_layout.addWidget(self.answer_display, stretch=4)

        transcript_panel = QWidget()
        transcript_panel.setObjectName("transcriptPanel")
        transcript_layout = QVBoxLayout(transcript_panel)
        transcript_layout.setContentsMargins(10, 8, 10, 8)
        transcript_layout.setSpacing(4)

        header = QHBoxLayout()
        header.addWidget(QLabel("🎙 NVIDIA Nemotron Live Transcript"))
        header.addStretch()
        self.status_led = QLabel("● IDLE")
        header.addWidget(self.status_led)
        transcript_layout.addLayout(header)

        self.partial_transcript_label = QLabel("")
        self.partial_transcript_label.setTextFormat(Qt.PlainText)
        self.partial_transcript_label.setWordWrap(True)
        transcript_layout.addWidget(self.partial_transcript_label)

        self.transcript_display = QTextBrowser()
        self.transcript_display.setText(
            "Transcript history will appear here once audio recording starts..."
        )
        transcript_layout.addWidget(self.transcript_display)
        container_layout.addWidget(transcript_panel, stretch=2)

        prompt_bar = QWidget()
        prompt_layout = QHBoxLayout(prompt_bar)
        prompt_layout.setContentsMargins(8, 6, 8, 6)
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Ask Gemini about the current practice context...")
        self.prompt_input.returnPressed.connect(self.send_custom_query)
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self.send_custom_query)
        prompt_layout.addWidget(self.prompt_input)
        prompt_layout.addWidget(send_btn)
        container_layout.addWidget(prompt_bar)

        control_bar = QWidget()
        control_bar.setObjectName("controlBar")
        control_layout = QHBoxLayout(control_bar)
        control_layout.setContentsMargins(10, 6, 10, 6)
        self.record_btn = QPushButton("🎤 Start Listening")
        self.record_btn.setObjectName("recordBtn")
        self.record_btn.clicked.connect(self.toggle_recording)
        clear_btn = QPushButton("🗑")
        clear_btn.clicked.connect(self.clear_context)
        control_layout.addWidget(self.record_btn)
        control_layout.addWidget(clear_btn)
        control_layout.addStretch()
        control_layout.addWidget(QSizeGrip(self), 0, Qt.AlignBottom | Qt.AlignRight)
        container_layout.addWidget(control_bar)

        self.update_ui_stylesheet()

    def update_ui_stylesheet(self):
        opacity = max(0.5, min(1.0, float(self.settings.get("window_opacity", 0.90))))
        alpha = int(opacity * 255)
        font_size = int(self.settings.get("font_size", 13))
        self.setStyleSheet(
            f"""
            QWidget {{ font-family: 'Segoe UI', Arial, sans-serif; color: #E2E8F0; }}
            QFrame#container {{ background-color: rgba(26,26,30,{alpha}); border: 1px solid #3182CE; border-radius: 8px; }}
            QWidget#titleBar, QWidget#transcriptPanel {{ background-color: rgba(15,23,42,150); }}
            QLabel {{ color: #CBD5E1; }}
            QTextBrowser {{ background-color: rgba(15,23,42,110); border: none; color: #E2E8F0; font-size: {font_size}px; padding: 10px; }}
            QLineEdit {{ background-color: #1E293B; border: 1px solid #475569; border-radius: 4px; padding: 7px; color: white; }}
            QPushButton {{ background-color: #3182CE; border: none; border-radius: 4px; padding: 7px 10px; color: white; font-weight: 600; }}
            QPushButton:hover {{ background-color: #4299E1; }}
            QPushButton#recordBtn {{ background-color: #DC2626; }}
            QPushButton#captureBtn {{ background-color: #059669; }}
            QPushButton#windowBtn {{ background-color: transparent; min-width: 26px; }}
            """
        )

    def title_bar_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def title_bar_mouse_move(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def showEvent(self, event):
        super().showEvent(event)
        self.apply_invisible_mode()

    def apply_invisible_mode(self):
        try:
            set_window_invisible_to_capture(
                int(self.winId()), self.settings.get("invisible_mode", True)
            )
        except Exception as exc:
            print(f"[overlay] Capture protection error: {exc}")

    @Slot()
    def toggle_recording(self):
        if self.audio_recorder.is_recording:
            self.audio_recorder.stop_recording()
            self.record_btn.setText("🎤 Start Listening")
            self.partial_transcript_label.clear()
            self._set_status("IDLE")
            return

        self.audio_recorder.set_devices(
            self.settings.get("mic_device_idx", -1),
            self.settings.get("system_device_idx", -1),
        )
        self.stt_worker.set_api_key(self.get_effective_nvidia_key())
        self.audio_recorder.start_recording()

        if not self.audio_recorder.mic_stream and not self.audio_recorder.system_stream:
            self.audio_recorder.stop_recording()
            QMessageBox.warning(
                self,
                "Audio Stream Error",
                "No audio input opened. Choose a microphone and/or system loopback device in Settings.",
            )
            return

        self.record_btn.setText("⏹ Stop Listening")
        self.transcript_display.setText("Listening…")
        self._set_status("LISTENING")

    @Slot(str, str)
    def handle_partial_transcription(self, speaker, text):
        self.partial_transcript_label.setText(f"Live · {speaker}: {text}")

    @Slot(str, str)
    def handle_transcription(self, speaker, text):
        self.partial_transcript_label.clear()
        self.copilot_ai.add_transcript_line(speaker, text)

        current = self.transcript_display.toPlainText()
        if current in {"Listening…", "Transcript history will appear here once audio recording starts..."}:
            self.transcript_display.clear()

        color = "#60A5FA" if speaker == "Candidate" else "#F59E0B"
        self.transcript_display.append(
            f'<b style="color:{color};">{speaker}:</b> {text}<br>'
        )
        self.transcript_display.moveCursor(QTextCursor.End)

        if speaker == "Interviewer" and len(text.strip()) > 8:
            now = time.monotonic()
            if now - self.last_query_time > 4.0:
                self.last_query_time = now
                self.current_trigger_source = f"🎙 Spoken question: *{text}*"
                self.trigger_text_analysis()

    @Slot(str)
    def update_status_log(self, status):
        print(f"[status] {status}")

    @Slot(str)
    def handle_stt_error(self, message):
        self.partial_transcript_label.clear()
        self.answer_display.setMarkdown(
            f"### NVIDIA Speech-to-Text Error\n\n`{message}`\n\n"
            "Check `NVIDIA_API_KEY`, the Riva endpoint, and your audio devices."
        )

    def _configure_gemini(self):
        self.copilot_ai.set_config(
            model=self.settings.get("model", config.DEFAULT_GEMINI_MODEL),
            api_key=self.get_effective_gemini_key(),
        )

    @Slot()
    def trigger_text_analysis(self):
        self._configure_gemini()
        self._set_status("THINKING")
        if not self.current_trigger_source:
            self.current_trigger_source = "🎙 Current transcript"
        self.answer_display.setMarkdown("### Thinking…\n\nGemini is analyzing the transcript.")
        self._start_ai_worker(custom_query=None)

    @Slot()
    def trigger_screen_analysis(self):
        self._configure_gemini()
        self._set_status("THINKING")
        if not self.current_trigger_source:
            self.current_trigger_source = "📸 Screen capture"

        try:
            image = capture_screen(self.settings.get("capture_region"))
            if image.width > 1280:
                from PIL import Image

                ratio = 1280.0 / image.width
                image = image.resize(
                    (1280, int(image.height * ratio)), Image.Resampling.LANCZOS
                )
            image_bytes = get_image_bytes(image, format="JPEG", quality=65)
        except Exception as exc:
            self._set_status("LISTENING" if self.audio_recorder.is_recording else "IDLE")
            self.answer_display.setMarkdown(f"### Screen Capture Failed\n\n`{exc}`")
            return

        self.answer_display.setMarkdown("### Thinking…\n\nGemini is analyzing the screen and transcript.")
        self._start_ai_worker(image_bytes=image_bytes)

    @Slot()
    def send_custom_query(self):
        query = self.prompt_input.text().strip()
        if not query:
            return
        self.prompt_input.clear()
        self._configure_gemini()
        self._set_status("THINKING")
        self.current_trigger_source = f"💬 Typed query: *{query}*"
        self.answer_display.setMarkdown("### Thinking…")
        self._start_ai_worker(custom_query=query)

    def _start_ai_worker(self, image_bytes=None, custom_query=None):
        if self.ai_worker and self.ai_worker.isRunning():
            return
        self.ai_worker = AIQueryWorker(
            self.copilot_ai,
            image_bytes=image_bytes,
            custom_query=custom_query,
        )
        self.ai_worker.answer_ready.connect(self.display_ai_answer)
        self.ai_worker.start()

    @Slot(str)
    def display_ai_answer(self, markdown_text):
        self._set_status("LISTENING" if self.audio_recorder.is_recording else "IDLE")
        context = self.current_trigger_source or "Current context"
        block = f"### Context / Question\n{context}\n\n---\n\n{markdown_text}"
        self.answer_history.insert(0, block)
        self.answer_history = self.answer_history[:20]
        self.answer_display.setMarkdown("\n\n***\n\n".join(self.answer_history))
        self.answer_display.verticalScrollBar().setValue(0)
        self.current_trigger_source = None

    def _set_status(self, state):
        styles = {
            "IDLE": "#94A3B8",
            "LISTENING": "#22C55E",
            "THINKING": "#60A5FA",
        }
        self.status_led.setText(f"● {state}")
        self.status_led.setStyleSheet(f"color: {styles.get(state, '#94A3B8')}; font-weight: bold;")

    @Slot()
    def clear_context(self):
        self.copilot_ai.clear_history()
        self.answer_history.clear()
        self.transcript_display.setText("[Context cleared]")
        self.partial_transcript_label.clear()
        self.answer_display.setMarkdown("### Context cleared")

    @Slot()
    def open_settings(self):
        dialog = SettingsDialog(self.settings, self)
        if not dialog.exec():
            return

        self.settings = dialog.settings.copy()
        self.update_ui_stylesheet()
        self._apply_window_flags()
        self.show()
        self.apply_invisible_mode()
        self.setup_global_hotkeys()

        self.audio_recorder.set_devices(
            self.settings.get("mic_device_idx", -1),
            self.settings.get("system_device_idx", -1),
        )
        self.copilot_ai.set_config(
            model=self.settings.get("model", config.DEFAULT_GEMINI_MODEL),
            api_key=self.get_effective_gemini_key(),
        )
        self.stt_worker.set_api_key(self.get_effective_nvidia_key())

    def closeEvent(self, event):
        if self.hotkey_listener:
            self.hotkey_listener.stop()
        if self.audio_recorder:
            self.audio_recorder.stop_recording()
        if self.stt_worker:
            self.stt_worker.stop()
        controller = getattr(self, "mouse_passthrough_controller", None)
        if controller:
            controller.stop()
        event.accept()
