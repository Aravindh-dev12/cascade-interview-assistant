import os
import time

from PySide6.QtCore import QObject, QPoint, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
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
        self.answer_ready.emit(
            self.copilot_ai.generate_answer(
                image_bytes=self.image_bytes,
                custom_query=self.custom_query,
            )
        )


def _practice_mode_enabled():
    return os.environ.get("PRACTICE_MODE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


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
        self.last_query_time = 0.0
        self.ai_worker = None
        self.ai_queue = []
        self.active_source = None

        self._apply_window_flags()
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMinimumSize(430, 520)
        self.resize(500, 720)

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
        return (
            os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        )

    def get_effective_nvidia_key(self):
        return os.environ.get("NVIDIA_API_KEY", "").strip()

    def setup_global_hotkeys(self):
        try:
            if self.hotkey_listener:
                self.hotkey_listener.stop()
            self.hotkey_listener = keyboard.GlobalHotKeys(
                {
                    self.settings.get("hotkey_capture", "<ctrl>+<shift>+s"):
                        lambda: self.hotkey_signaler.capture_hotkey_triggered.emit(),
                    self.settings.get("hotkey_record", "<ctrl>+<shift>+a"):
                        lambda: self.hotkey_signaler.record_hotkey_triggered.emit(),
                }
            )
            self.hotkey_listener.start()
        except Exception as exc:
            print(f"[hotkey] Error registering global hotkeys: {exc}")

    def init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(0)

        self.container = QFrame(self)
        self.container.setObjectName("container")
        root.addWidget(self.container)
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        title_bar = QWidget()
        title_bar.setObjectName("titleBar")
        title_bar.setProperty("overlayInteractive", True)
        title_bar.setFixedHeight(58)
        title_row = QHBoxLayout(title_bar)
        title_row.setContentsMargins(14, 0, 10, 0)
        title_row.setSpacing(9)

        mark = QLabel("Q")
        mark.setObjectName("brandMark")
        mark.setFixedSize(32, 32)
        mark.setAlignment(Qt.AlignCenter)
        brand = QVBoxLayout()
        brand.setSpacing(0)
        name = QLabel("quntumnintent")
        name.setObjectName("brandTitle")
        subtitle = QLabel("live voice · chat · vision")
        subtitle.setObjectName("brandSubtitle")
        brand.addWidget(name)
        brand.addWidget(subtitle)

        self.status_badge = QLabel("IDLE")
        self.status_badge.setObjectName("statusBadge")
        self.status_badge.setAlignment(Qt.AlignCenter)
        self.status_badge.setMinimumWidth(78)

        settings_btn = QPushButton("⚙")
        settings_btn.setObjectName("iconButton")
        settings_btn.setFixedWidth(34)
        settings_btn.clicked.connect(self.open_settings)
        min_btn = QPushButton("—")
        min_btn.setObjectName("iconButton")
        min_btn.setFixedWidth(34)
        min_btn.clicked.connect(self.showMinimized)
        close_btn = QPushButton("×")
        close_btn.setObjectName("closeButton")
        close_btn.setFixedWidth(34)
        close_btn.clicked.connect(self.close)

        title_row.addWidget(mark)
        title_row.addLayout(brand)
        title_row.addStretch()
        title_row.addWidget(self.status_badge)
        title_row.addWidget(settings_btn)
        title_row.addWidget(min_btn)
        title_row.addWidget(close_btn)
        title_bar.mousePressEvent = self.title_bar_mouse_press
        title_bar.mouseMoveEvent = self.title_bar_mouse_move
        layout.addWidget(title_bar)

        body = QWidget()
        body.setObjectName("body")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(12, 12, 12, 12)
        body_layout.setSpacing(10)

        header = QHBoxLayout()
        assistant_label = QLabel("ANSWER")
        assistant_label.setObjectName("eyebrow")
        self.mode_label = QLabel(self.settings.get("model", config.DEFAULT_GEMINI_MODEL))
        self.mode_label.setObjectName("muted")
        header.addWidget(assistant_label)
        header.addStretch()
        header.addWidget(self.mode_label)
        body_layout.addLayout(header)

        answer_card = QFrame()
        answer_card.setObjectName("card")
        answer_layout = QVBoxLayout(answer_card)
        answer_layout.setContentsMargins(0, 0, 0, 0)
        self.answer_display = QTextBrowser()
        self.answer_display.setObjectName("answerDisplay")
        self.answer_display.setOpenExternalLinks(True)
        self.answer_display.setMarkdown(
            "### Ready\n\n"
            "Speak in practice mode, type a question, or capture the screen. "
            "API credentials are loaded automatically from `.env`."
        )
        answer_layout.addWidget(self.answer_display)
        body_layout.addWidget(answer_card, stretch=5)

        transcript_header = QHBoxLayout()
        transcript_title = QLabel("LIVE TRANSCRIPT")
        transcript_title.setObjectName("eyebrow")
        transcript_meta = QLabel("NVIDIA NEMOTRON")
        transcript_meta.setObjectName("muted")
        transcript_header.addWidget(transcript_title)
        transcript_header.addStretch()
        transcript_header.addWidget(transcript_meta)
        body_layout.addLayout(transcript_header)

        transcript_card = QFrame()
        transcript_card.setObjectName("card")
        transcript_layout = QVBoxLayout(transcript_card)
        transcript_layout.setContentsMargins(10, 8, 10, 8)
        transcript_layout.setSpacing(6)
        self.partial_transcript_label = QLabel("")
        self.partial_transcript_label.setObjectName("partialTranscript")
        self.partial_transcript_label.setWordWrap(True)
        self.partial_transcript_label.hide()
        self.transcript_display = QTextBrowser()
        self.transcript_display.setObjectName("transcriptDisplay")
        self.transcript_display.setText("Listening starts automatically when enabled.")
        transcript_layout.addWidget(self.partial_transcript_label)
        transcript_layout.addWidget(self.transcript_display)
        body_layout.addWidget(transcript_card, stretch=2)

        composer = QFrame()
        composer.setObjectName("composer")
        composer_row = QHBoxLayout(composer)
        composer_row.setContentsMargins(9, 7, 7, 7)
        composer_row.setSpacing(7)
        self.prompt_input = QLineEdit()
        self.prompt_input.setObjectName("promptInput")
        self.prompt_input.setPlaceholderText("Ask anything about the current context…")
        self.prompt_input.returnPressed.connect(self.send_custom_query)
        send_btn = QPushButton("Send")
        send_btn.setObjectName("primaryButton")
        send_btn.clicked.connect(self.send_custom_query)
        composer_row.addWidget(self.prompt_input, stretch=1)
        composer_row.addWidget(send_btn)
        body_layout.addWidget(composer)

        control_bar = QFrame()
        control_bar.setObjectName("controlBar")
        control_row = QHBoxLayout(control_bar)
        control_row.setContentsMargins(0, 0, 0, 0)
        control_row.setSpacing(7)
        self.record_btn = QPushButton("Listen")
        self.record_btn.setObjectName("recordBtn")
        self.record_btn.clicked.connect(self.toggle_recording)
        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("secondaryButton")
        clear_btn.clicked.connect(self.clear_context)
        control_row.addWidget(self.record_btn)
        control_row.addWidget(clear_btn)
        control_row.addStretch()
        control_row.addWidget(QSizeGrip(self), 0, Qt.AlignBottom | Qt.AlignRight)
        body_layout.addWidget(control_bar)

        layout.addWidget(body)
        self.update_ui_stylesheet()
        self._set_status("IDLE")

    def update_ui_stylesheet(self):
        opacity = max(0.55, min(1.0, float(self.settings.get("window_opacity", 0.94))))
        alpha = int(opacity * 255)
        font_size = int(self.settings.get("font_size", 13))
        self.setStyleSheet(f"""
            QWidget {{ font-family:'Segoe UI Variable Text','Segoe UI',Arial,sans-serif; color:#E8EEF8; font-size:13px; }}
            QFrame#container {{ background:rgba(7,11,18,{alpha}); border:1px solid rgba(71,85,105,150); border-radius:14px; }}
            QWidget#titleBar {{ background:rgba(10,16,27,240); border-bottom:1px solid rgba(51,65,85,170); border-top-left-radius:14px; border-top-right-radius:14px; }}
            QWidget#body {{ background:transparent; }}
            QLabel#brandMark {{ background:#2563EB; color:white; border-radius:9px; font-weight:800; }}
            QLabel#brandTitle {{ color:#F8FAFC; font-size:14px; font-weight:750; }}
            QLabel#brandSubtitle, QLabel#muted {{ color:#718096; font-size:10px; }}
            QLabel#eyebrow {{ color:#93A4BA; font-size:10px; font-weight:800; }}
            QFrame#card, QFrame#composer {{ background:rgba(15,23,42,195); border:1px solid rgba(51,65,85,185); border-radius:10px; }}
            QTextBrowser#answerDisplay {{ background:transparent; border:none; color:#E8EEF8; padding:13px; font-size:{font_size}px; }}
            QTextBrowser#transcriptDisplay {{ background:transparent; border:none; color:#B7C4D5; font-size:11px; }}
            QLabel#partialTranscript {{ color:#BFDBFE; background:rgba(30,64,175,75); border:1px solid rgba(59,130,246,105); border-radius:7px; padding:6px 8px; font-size:11px; }}
            QLineEdit#promptInput {{ background:transparent; border:none; color:#F8FAFC; padding:7px 3px; }}
            QPushButton {{ min-height:32px; border-radius:7px; padding:0 11px; font-weight:650; }}
            QPushButton#primaryButton {{ background:#2563EB; color:white; border:1px solid #3B82F6; }}
            QPushButton#recordBtn {{ background:#F8FAFC; color:#0F172A; border:1px solid #E2E8F0; min-width:78px; }}
            QPushButton#secondaryButton, QPushButton#captureBtn, QPushButton#iconButton {{ background:transparent; color:#B1BED0; border:1px solid #334155; }}
            QPushButton#secondaryButton:hover, QPushButton#captureBtn:hover, QPushButton#iconButton:hover {{ background:#1E293B; color:white; }}
            QPushButton#closeButton {{ background:transparent; color:#A8B6C8; border:1px solid transparent; font-size:16px; }}
            QPushButton#closeButton:hover {{ background:#7F1D1D; color:white; border-color:#991B1B; }}
            QScrollBar:vertical {{ background:transparent; width:7px; }}
            QScrollBar::handle:vertical {{ background:#334155; border-radius:3px; min-height:28px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
        """)

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
            self.record_btn.setText("Listen")
            self.partial_transcript_label.clear()
            self.partial_transcript_label.hide()
            self._set_status("IDLE")
            return

        if not self.get_effective_nvidia_key():
            QMessageBox.warning(self, "NVIDIA key missing", "Add NVIDIA_API_KEY to the project .env and restart the app.")
            return

        self.audio_recorder.set_devices(
            self.settings.get("mic_device_idx", -1),
            self.settings.get("system_device_idx", -1),
        )
        self.stt_worker.set_api_key(self.get_effective_nvidia_key())
        self.audio_recorder.start_recording()
        if not self.audio_recorder.mic_stream and not self.audio_recorder.system_stream:
            self.audio_recorder.stop_recording()
            QMessageBox.warning(self, "Audio input unavailable", "No audio input could be opened. Choose a microphone or system loopback device in Settings.")
            return

        self.record_btn.setText("Stop")
        self.transcript_display.setText("Listening for speech…")
        self._set_status("LISTENING")

    @Slot(str, str)
    def handle_partial_transcription(self, speaker, text):
        self.partial_transcript_label.setText(f"{speaker} · {text}")
        self.partial_transcript_label.show()

    @Slot(str, str)
    def handle_transcription(self, speaker, text):
        self.partial_transcript_label.clear()
        self.partial_transcript_label.hide()
        self.copilot_ai.add_transcript_line(speaker, text)

        if self.transcript_display.toPlainText() in {
            "Listening for speech…", "Listening starts automatically when enabled."
        }:
            self.transcript_display.clear()
        color = "#60A5FA" if speaker == "Candidate" else "#FBBF24"
        self.transcript_display.append(
            f'<b style="color:{color};">{speaker}</b>'
            f'<span style="color:#64748B;"> · </span>{text}<br>'
        )
        self.transcript_display.moveCursor(QTextCursor.End)

        if (
            speaker == "Interviewer"
            and len(text.strip()) > 8
            and self.settings.get("auto_answer_speech", True)
            and _practice_mode_enabled()
        ):
            cooldown = float(self.settings.get("answer_cooldown_seconds", 1.2))
            now = time.monotonic()
            if now - self.last_query_time >= cooldown:
                self.last_query_time = now
                self._enqueue_ai(
                    source=f"Spoken question: *{text}*",
                    custom_query=None,
                )

    @Slot(str)
    def update_status_log(self, status):
        print(f"[status] {status}")

    @Slot(str)
    def handle_stt_error(self, message):
        self.partial_transcript_label.hide()
        self._set_status("ERROR")
        self.answer_display.setMarkdown(f"### Voice transcription unavailable\n\n`{message}`")

    def _configure_gemini(self):
        self.copilot_ai.set_config(
            model=self.settings.get("model", config.DEFAULT_GEMINI_MODEL),
            api_key=self.get_effective_gemini_key(),
        )

    @Slot()
    def trigger_text_analysis(self):
        self._enqueue_ai(source="Current transcript", custom_query=None)

    @Slot()
    def trigger_screen_analysis(self):
        self._configure_gemini()
        try:
            image = capture_screen(self.settings.get("capture_region"))
            if image.width > 1280:
                from PIL import Image
                ratio = 1280.0 / image.width
                image = image.resize((1280, int(image.height * ratio)), Image.Resampling.LANCZOS)
            image_bytes = get_image_bytes(image, format="JPEG", quality=70)
        except Exception as exc:
            self.answer_display.setMarkdown(f"### Screen capture failed\n\n`{exc}`")
            return
        self._enqueue_ai(source="Screen capture", image_bytes=image_bytes)

    @Slot()
    def send_custom_query(self):
        query = self.prompt_input.text().strip()
        if not query:
            return
        self.prompt_input.clear()
        self._enqueue_ai(source=f"Chat: *{query}*", custom_query=query)

    def _enqueue_ai(self, source, image_bytes=None, custom_query=None):
        self._configure_gemini()
        if not self.get_effective_gemini_key():
            self.answer_display.setMarkdown("### Gemini key missing\n\nAdd `GEMINI_API_KEY` to the project `.env` and restart the app.")
            self._set_status("ERROR")
            return
        self.ai_queue.append((source, image_bytes, custom_query))
        self._run_next_ai()

    def _run_next_ai(self):
        if self.ai_worker and self.ai_worker.isRunning():
            self._set_status("THINKING")
            return
        if not self.ai_queue:
            self._set_status("LISTENING" if self.audio_recorder.is_recording else "IDLE")
            return

        source, image_bytes, custom_query = self.ai_queue.pop(0)
        self.active_source = source
        self._set_status("THINKING")
        self.answer_display.setMarkdown(f"### Thinking…\n\n{source}")
        self.ai_worker = AIQueryWorker(
            self.copilot_ai,
            image_bytes=image_bytes,
            custom_query=custom_query,
        )
        self.ai_worker.answer_ready.connect(self.display_ai_answer)
        self.ai_worker.finished.connect(self._ai_worker_finished)
        self.ai_worker.start()

    @Slot(str)
    def display_ai_answer(self, markdown_text):
        source = self.active_source or "Current context"
        block = f"### {source}\n\n{markdown_text}"
        self.answer_history.insert(0, block)
        self.answer_history = self.answer_history[:16]
        self.answer_display.setMarkdown("\n\n---\n\n".join(self.answer_history))
        self.answer_display.verticalScrollBar().setValue(0)

    @Slot()
    def _ai_worker_finished(self):
        worker = self.ai_worker
        self.ai_worker = None
        self.active_source = None
        if worker:
            worker.deleteLater()
        QTimer.singleShot(0, self._run_next_ai)

    def _set_status(self, state):
        palette = {
            "IDLE": ("#94A3B8", "rgba(30,41,59,190)", "#475569"),
            "LISTENING": ("#86EFAC", "rgba(20,83,45,180)", "#166534"),
            "THINKING": ("#93C5FD", "rgba(30,64,175,160)", "#1D4ED8"),
            "ERROR": ("#FCA5A5", "rgba(127,29,29,170)", "#991B1B"),
        }
        fg, bg, border = palette.get(state, palette["IDLE"])
        self.status_badge.setText(state)
        self.status_badge.setStyleSheet(
            f"color:{fg}; background:{bg}; border:1px solid {border}; border-radius:10px; padding:5px 9px; font-size:10px; font-weight:800;"
        )

    @Slot()
    def clear_context(self):
        self.copilot_ai.clear_history()
        self.answer_history.clear()
        self.ai_queue.clear()
        self.transcript_display.setText("Transcript context cleared.")
        self.partial_transcript_label.hide()
        self.answer_display.setMarkdown("### Cleared\n\nReady for a new session.")

    @Slot()
    def open_settings(self):
        dialog = SettingsDialog(self.settings, self)
        if not dialog.exec():
            return

        was_recording = self.audio_recorder.is_recording
        self.settings = dialog.settings.copy()
        self.mode_label.setText(self.settings.get("model", config.DEFAULT_GEMINI_MODEL))
        self.update_ui_stylesheet()
        self._apply_window_flags()
        self.show()
        self.apply_invisible_mode()
        self.setup_global_hotkeys()
        self.audio_recorder.set_devices(
            self.settings.get("mic_device_idx", -1),
            self.settings.get("system_device_idx", -1),
        )
        self._configure_gemini()
        self.stt_worker.set_api_key(self.get_effective_nvidia_key())

        if (
            not was_recording
            and self.settings.get("auto_start_listening", True)
            and _practice_mode_enabled()
            and self.get_effective_nvidia_key()
        ):
            QTimer.singleShot(200, self.toggle_recording)

    def closeEvent(self, event):
        if self.hotkey_listener:
            self.hotkey_listener.stop()
        self.audio_recorder.stop_recording()
        self.stt_worker.stop()
        controller = getattr(self, "mouse_passthrough_controller", None)
        if controller:
            controller.stop()
        event.accept()
