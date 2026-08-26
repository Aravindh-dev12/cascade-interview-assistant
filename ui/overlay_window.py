import os
import time

from PySide6.QtCore import QObject, QPoint, QThread, Qt, Signal, Slot
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
        self.setMinimumSize(440, 560)
        self.resize(520, 780)

        self.init_ui()
        self.setup_global_hotkeys()

        self.stt_worker.partial_transcription_ready.connect(
            self.handle_partial_transcription
        )
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

    def _section_header(self, eyebrow, detail):
        row = QWidget()
        row.setObjectName("sectionHeader")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QLabel(eyebrow)
        title.setObjectName("eyebrow")
        detail_label = QLabel(detail)
        detail_label.setObjectName("sectionMeta")

        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(detail_label)
        return row

    def init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(0)

        self.container = QFrame(self)
        self.container.setObjectName("container")
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        root_layout.addWidget(self.container)

        title_bar = QWidget()
        title_bar.setObjectName("titleBar")
        title_bar.setProperty("overlayInteractive", True)
        title_bar.setFixedHeight(64)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(16, 0, 10, 0)
        title_layout.setSpacing(10)

        brand_mark = QLabel("Q")
        brand_mark.setObjectName("brandMark")
        brand_mark.setFixedSize(34, 34)
        brand_mark.setAlignment(Qt.AlignCenter)

        brand_copy = QVBoxLayout()
        brand_copy.setContentsMargins(0, 0, 0, 0)
        brand_copy.setSpacing(1)
        brand_title = QLabel("quntumnintent")
        brand_title.setObjectName("brandTitle")
        brand_subtitle = QLabel("Gemini reasoning · NVIDIA voice")
        brand_subtitle.setObjectName("brandSubtitle")
        brand_copy.addWidget(brand_title)
        brand_copy.addWidget(brand_subtitle)

        self.status_badge = QLabel("IDLE")
        self.status_badge.setObjectName("statusBadge")
        self.status_badge.setAlignment(Qt.AlignCenter)
        self.status_badge.setMinimumWidth(82)

        settings_btn = QPushButton("Settings")
        settings_btn.setObjectName("titleAction")
        settings_btn.clicked.connect(self.open_settings)

        min_btn = QPushButton("—")
        min_btn.setObjectName("windowControl")
        min_btn.setFixedWidth(34)
        min_btn.clicked.connect(self.showMinimized)

        close_btn = QPushButton("×")
        close_btn.setObjectName("closeControl")
        close_btn.setFixedWidth(34)
        close_btn.clicked.connect(self.close)

        title_layout.addWidget(brand_mark)
        title_layout.addLayout(brand_copy)
        title_layout.addStretch()
        title_layout.addWidget(self.status_badge)
        title_layout.addWidget(settings_btn)
        title_layout.addWidget(min_btn)
        title_layout.addWidget(close_btn)

        title_bar.mousePressEvent = self.title_bar_mouse_press
        title_bar.mouseMoveEvent = self.title_bar_mouse_move
        container_layout.addWidget(title_bar)

        body = QWidget()
        body.setObjectName("body")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(14, 14, 14, 14)
        body_layout.setSpacing(12)

        body_layout.addWidget(
            self._section_header(
                "ASSISTANT OUTPUT",
                self.settings.get("model", config.DEFAULT_GEMINI_MODEL),
            )
        )

        answer_card = QFrame()
        answer_card.setObjectName("answerCard")
        answer_layout = QVBoxLayout(answer_card)
        answer_layout.setContentsMargins(0, 0, 0, 0)
        answer_layout.setSpacing(0)

        self.answer_display = QTextBrowser()
        self.answer_display.setObjectName("answerDisplay")
        self.answer_display.setOpenExternalLinks(True)
        self.answer_display.setMarkdown(
            "### Ready for practice\n\n"
            "Start listening to build transcript context, capture the configured screen "
            "region, or type a question below. Gemini handles reasoning and NVIDIA "
            "Nemotron provides real-time transcription."
        )
        answer_layout.addWidget(self.answer_display)
        body_layout.addWidget(answer_card, stretch=5)

        body_layout.addWidget(
            self._section_header("LIVE TRANSCRIPT", "NVIDIA NEMOTRON")
        )

        transcript_card = QFrame()
        transcript_card.setObjectName("transcriptCard")
        transcript_layout = QVBoxLayout(transcript_card)
        transcript_layout.setContentsMargins(12, 10, 12, 10)
        transcript_layout.setSpacing(8)

        self.partial_transcript_label = QLabel("")
        self.partial_transcript_label.setObjectName("partialTranscript")
        self.partial_transcript_label.setTextFormat(Qt.PlainText)
        self.partial_transcript_label.setWordWrap(True)
        self.partial_transcript_label.hide()
        transcript_layout.addWidget(self.partial_transcript_label)

        self.transcript_display = QTextBrowser()
        self.transcript_display.setObjectName("transcriptDisplay")
        self.transcript_display.setText(
            "Transcript history will appear here when listening starts."
        )
        transcript_layout.addWidget(self.transcript_display)
        body_layout.addWidget(transcript_card, stretch=2)

        composer = QFrame()
        composer.setObjectName("composer")
        prompt_layout = QHBoxLayout(composer)
        prompt_layout.setContentsMargins(10, 8, 8, 8)
        prompt_layout.setSpacing(8)

        self.prompt_input = QLineEdit()
        self.prompt_input.setObjectName("promptInput")
        self.prompt_input.setPlaceholderText("Ask Gemini about the current context")
        self.prompt_input.returnPressed.connect(self.send_custom_query)

        send_btn = QPushButton("Send")
        send_btn.setObjectName("primaryButton")
        send_btn.clicked.connect(self.send_custom_query)
        prompt_layout.addWidget(self.prompt_input, stretch=1)
        prompt_layout.addWidget(send_btn)
        body_layout.addWidget(composer)

        control_bar = QFrame()
        control_bar.setObjectName("controlBar")
        control_layout = QHBoxLayout(control_bar)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(8)

        self.record_btn = QPushButton("Start listening")
        self.record_btn.setObjectName("recordBtn")
        self.record_btn.clicked.connect(self.toggle_recording)

        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("secondaryButton")
        clear_btn.clicked.connect(self.clear_context)

        control_layout.addWidget(self.record_btn)
        control_layout.addWidget(clear_btn)
        control_layout.addStretch()
        control_layout.addWidget(QSizeGrip(self), 0, Qt.AlignBottom | Qt.AlignRight)
        body_layout.addWidget(control_bar)

        container_layout.addWidget(body)
        self.update_ui_stylesheet()
        self._set_status("IDLE")

    def update_ui_stylesheet(self):
        opacity = max(
            0.5, min(1.0, float(self.settings.get("window_opacity", 0.94)))
        )
        alpha = int(opacity * 255)
        font_size = int(self.settings.get("font_size", 13))

        self.setStyleSheet(
            f"""
            QWidget {{
                font-family: 'Segoe UI Variable Text', 'Segoe UI', Arial, sans-serif;
                color: #E6EDF7;
                font-size: 13px;
            }}
            QFrame#container {{
                background-color: rgba(9, 14, 25, {alpha});
                border: 1px solid rgba(71, 85, 105, 145);
                border-radius: 14px;
            }}
            QWidget#titleBar {{
                background-color: rgba(12, 18, 31, 235);
                border-bottom: 1px solid rgba(51, 65, 85, 170);
                border-top-left-radius: 14px;
                border-top-right-radius: 14px;
            }}
            QWidget#body {{ background: transparent; }}
            QLabel#brandMark {{
                background-color: #2563EB;
                color: white;
                border-radius: 9px;
                font-size: 15px;
                font-weight: 800;
            }}
            QLabel#brandTitle {{ color: #F8FAFC; font-size: 14px; font-weight: 700; }}
            QLabel#brandSubtitle {{ color: #7F8EA3; font-size: 10px; font-weight: 500; }}
            QLabel#statusBadge {{
                color: #94A3B8;
                background-color: rgba(30, 41, 59, 180);
                border: 1px solid rgba(71, 85, 105, 190);
                border-radius: 10px;
                padding: 5px 10px;
                font-size: 10px;
                font-weight: 800;
            }}
            QLabel#eyebrow {{ color: #8EA0B8; font-size: 10px; font-weight: 800; }}
            QLabel#sectionMeta {{ color: #5F7188; font-size: 10px; font-weight: 600; }}
            QFrame#answerCard,
            QFrame#transcriptCard,
            QFrame#composer {{
                background-color: rgba(15, 23, 42, 190);
                border: 1px solid rgba(51, 65, 85, 185);
                border-radius: 10px;
            }}
            QTextBrowser#answerDisplay {{
                background: transparent;
                border: none;
                color: #E6EDF7;
                padding: 14px;
                font-size: {font_size}px;
                selection-background-color: #1D4ED8;
            }}
            QTextBrowser#transcriptDisplay {{
                background: transparent;
                border: none;
                color: #B8C5D6;
                padding: 0;
                font-size: 11px;
                selection-background-color: #1D4ED8;
            }}
            QLabel#partialTranscript {{
                color: #BFDBFE;
                background-color: rgba(30, 64, 175, 80);
                border: 1px solid rgba(59, 130, 246, 100);
                border-radius: 7px;
                padding: 7px 9px;
                font-size: 11px;
            }}
            QLineEdit#promptInput {{
                background: transparent;
                border: none;
                color: #F8FAFC;
                padding: 7px 4px;
                font-size: 12px;
            }}
            QLineEdit#promptInput:focus {{ border: none; }}
            QPushButton {{
                min-height: 32px;
                border-radius: 7px;
                padding: 0 12px;
                font-weight: 650;
            }}
            QPushButton#primaryButton {{
                background-color: #2563EB;
                border: 1px solid #3B82F6;
                color: white;
            }}
            QPushButton#primaryButton:hover {{ background-color: #1D4ED8; }}
            QPushButton#recordBtn {{
                background-color: #F8FAFC;
                color: #0F172A;
                border: 1px solid #E2E8F0;
                min-width: 118px;
            }}
            QPushButton#recordBtn:hover {{ background-color: #E2E8F0; }}
            QPushButton#captureBtn {{
                background-color: #172033;
                color: #D7E2F0;
                border: 1px solid #334155;
                min-width: 112px;
            }}
            QPushButton#captureBtn:hover,
            QPushButton#secondaryButton:hover,
            QPushButton#titleAction:hover,
            QPushButton#windowControl:hover {{
                background-color: #1E293B;
                border-color: #475569;
                color: #F8FAFC;
            }}
            QPushButton#secondaryButton,
            QPushButton#titleAction,
            QPushButton#windowControl {{
                background-color: transparent;
                color: #A8B6C8;
                border: 1px solid #334155;
            }}
            QPushButton#titleAction {{ min-height: 28px; padding: 0 10px; font-size: 11px; }}
            QPushButton#windowControl,
            QPushButton#closeControl {{
                min-height: 28px;
                padding: 0;
                font-size: 16px;
                font-weight: 500;
            }}
            QPushButton#closeControl {{
                background-color: transparent;
                color: #A8B6C8;
                border: 1px solid transparent;
            }}
            QPushButton#closeControl:hover {{
                background-color: #7F1D1D;
                border-color: #991B1B;
                color: white;
            }}
            QScrollBar:vertical {{ background: transparent; width: 8px; margin: 4px 2px 4px 0; }}
            QScrollBar::handle:vertical {{ background: #334155; border-radius: 4px; min-height: 28px; }}
            QScrollBar::handle:vertical:hover {{ background: #475569; }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{ height: 0; }}
            QSizeGrip {{ width: 12px; height: 12px; background: transparent; }}
            """
        )

    def title_bar_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
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
            self.record_btn.setText("Start listening")
            self.partial_transcript_label.clear()
            self.partial_transcript_label.hide()
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
                "Audio input unavailable",
                "No audio input could be opened. Choose a microphone and/or "
                "system loopback device in Settings.",
            )
            return

        self.record_btn.setText("Stop listening")
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

        current = self.transcript_display.toPlainText()
        if current in {
            "Listening for speech…",
            "Transcript history will appear here when listening starts.",
        }:
            self.transcript_display.clear()

        color = "#60A5FA" if speaker == "Candidate" else "#FBBF24"
        self.transcript_display.append(
            f'<b style="color:{color};">{speaker}</b>'
            f'<span style="color:#64748B;">  ·  </span>{text}<br>'
        )
        self.transcript_display.moveCursor(QTextCursor.End)

        if speaker == "Interviewer" and len(text.strip()) > 8:
            now = time.monotonic()
            if now - self.last_query_time > 4.0:
                self.last_query_time = now
                self.current_trigger_source = f"Spoken question: *{text}*"
                self.trigger_text_analysis()

    @Slot(str)
    def update_status_log(self, status):
        print(f"[status] {status}")

    @Slot(str)
    def handle_stt_error(self, message):
        self.partial_transcript_label.clear()
        self.partial_transcript_label.hide()
        self._set_status("ERROR")
        self.answer_display.setMarkdown(
            "### Voice transcription unavailable\n\n"
            f"`{message}`\n\n"
            "Check `NVIDIA_API_KEY`, the Riva endpoint, and the selected audio devices."
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
            self.current_trigger_source = "Current transcript"
        self.answer_display.setMarkdown(
            "### Analyzing transcript\n\nGemini is preparing a concise response."
        )
        self._start_ai_worker(custom_query=None)

    @Slot()
    def trigger_screen_analysis(self):
        self._configure_gemini()
        self._set_status("THINKING")
        if not self.current_trigger_source:
            self.current_trigger_source = "Screen capture"

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
            self._set_status(
                "LISTENING" if self.audio_recorder.is_recording else "IDLE"
            )
            self.answer_display.setMarkdown(f"### Screen capture failed\n\n`{exc}`")
            return

        self.answer_display.setMarkdown(
            "### Analyzing screen\n\nGemini is reviewing the capture with transcript context."
        )
        self._start_ai_worker(image_bytes=image_bytes)

    @Slot()
    def send_custom_query(self):
        query = self.prompt_input.text().strip()
        if not query:
            return
        self.prompt_input.clear()
        self._configure_gemini()
        self._set_status("THINKING")
        self.current_trigger_source = f"Typed query: *{query}*"
        self.answer_display.setMarkdown(
            "### Processing request\n\nGemini is working on your instruction."
        )
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
        self._set_status(
            "LISTENING" if self.audio_recorder.is_recording else "IDLE"
        )
        context = self.current_trigger_source or "Current context"
        block = f"### Context\n{context}\n\n---\n\n{markdown_text}"
        self.answer_history.insert(0, block)
        self.answer_history = self.answer_history[:20]
        self.answer_display.setMarkdown("\n\n***\n\n".join(self.answer_history))
        self.answer_display.verticalScrollBar().setValue(0)
        self.current_trigger_source = None

    def _set_status(self, state):
        palette = {
            "IDLE": ("#94A3B8", "rgba(30, 41, 59, 190)", "#475569"),
            "LISTENING": ("#86EFAC", "rgba(20, 83, 45, 180)", "#166534"),
            "THINKING": ("#93C5FD", "rgba(30, 64, 175, 160)", "#1D4ED8"),
            "ERROR": ("#FCA5A5", "rgba(127, 29, 29, 170)", "#991B1B"),
        }
        fg, bg, border = palette.get(state, palette["IDLE"])
        self.status_badge.setText(state)
        self.status_badge.setStyleSheet(
            f"color:{fg}; background-color:{bg}; border:1px solid {border}; "
            "border-radius:10px; padding:5px 10px; font-size:10px; "
            "font-weight:800;"
        )

    @Slot()
    def clear_context(self):
        self.copilot_ai.clear_history()
        self.answer_history.clear()
        self.transcript_display.setText("Transcript context cleared.")
        self.partial_transcript_label.clear()
        self.partial_transcript_label.hide()
        self.answer_display.setMarkdown(
            "### Context cleared\n\nReady for a new practice session."
        )

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
