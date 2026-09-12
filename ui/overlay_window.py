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
from engine.question_detector import is_substantive_question, should_attach_screen
from engine.screen_watcher import ScreenWatcher
from engine.stt_worker import STTWorker
from ui.settings_dialog import SettingsDialog
from utils.win_utils import set_window_invisible_to_capture


def _practice_mode_enabled():
    return os.environ.get("PRACTICE_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}


class HotkeySignaler(QObject):
    capture_hotkey_triggered = Signal()
    record_hotkey_triggered = Signal()


class AIQueryWorker(QThread):
    chunk_ready = Signal(int, str)
    answer_ready = Signal(int, str)

    def __init__(self, request_id, copilot_ai, image_bytes=None, custom_query=None, use_image_history=False):
        super().__init__()
        self.request_id = int(request_id)
        self.copilot_ai = copilot_ai
        self.image_bytes = image_bytes
        self.custom_query = custom_query
        self.use_image_history = use_image_history

    def run(self):
        pieces = []
        for chunk in self.copilot_ai.generate_answer_stream(
            image_bytes=self.image_bytes,
            custom_query=self.custom_query,
            use_image_history=self.use_image_history,
        ):
            if self.isInterruptionRequested():
                break
            pieces.append(chunk)
            self.chunk_ready.emit(self.request_id, chunk)
        self.answer_ready.emit(self.request_id, "".join(pieces).strip())


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
        self.stt_worker = STTWorker(self.audio_recorder, api_key=self.get_effective_nvidia_key())
        self.copilot_ai = CopilotAI(provider=self.settings.get("ai_provider", "gemini"))

        self.drag_position = QPoint()
        self.hotkey_signaler = HotkeySignaler()
        self.hotkey_signaler.record_hotkey_triggered.connect(self.toggle_recording)
        self.hotkey_listener = None

        self.answer_history = []
        self.request_queue = []
        self.workers = {}
        self.requests = {}
        self.stale_request_ids = set()
        self.next_request_id = 1
        self.session_generation = 1
        self.streaming_text = {}
        self.latest_screen_bytes = None
        self.latest_screen_time = 0.0
        self.last_screen_answer_time = 0.0
        self.last_query_time = 0.0
        self.last_interviewer_time = 0.0

        self._apply_window_flags()
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMinimumSize(470, 520)
        self.resize(560, 750)
        self.init_ui()
        self.setup_global_hotkeys()

        self.stt_worker.partial_transcription_ready.connect(self.handle_partial_transcription)
        self.stt_worker.transcription_ready.connect(self.handle_transcription)
        self.stt_worker.status_updated.connect(self.update_status_log)
        self.stt_worker.error_occurred.connect(self.handle_stt_error)
        self.stt_worker.start()

        self.screen_watcher = ScreenWatcher(
            interval_ms=self.settings.get("screen_watch_interval_ms", 650),
            stable_ms=self.settings.get("screen_stable_ms", 450),
            change_threshold=self.settings.get("screen_change_threshold", 0.055),
            parent=self,
        )
        self.screen_watcher.frame_ready.connect(self.handle_screen_frame)
        self.screen_watcher.status_updated.connect(self.update_status_log)
        self.screen_watcher.error_occurred.connect(self.update_status_log)
        self._sync_screen_watcher(restart=True)

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

    def get_effective_nvidia_key(self):
        return os.environ.get("NVIDIA_API_KEY", "").strip()

    def _configure_ai(self):
        self.copilot_ai.set_config(provider=self.settings.get("ai_provider", "gemini"))

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
        subtitle = QLabel("real-time voice · screen · answers")
        subtitle.setObjectName("brandSubtitle")
        brand.addWidget(name)
        brand.addWidget(subtitle)

        self.status_badge = QLabel("IDLE")
        self.status_badge.setObjectName("statusBadge")
        self.status_badge.setAlignment(Qt.AlignCenter)
        self.status_badge.setMinimumWidth(82)

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
        self.mode_label = QLabel(self.copilot_ai.runtime_label())
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
            "Click Listen for NVIDIA Parakeet transcription, Capture screen for visual questions, or Send to ask Gemini 2.5 Flash."
        )
        answer_layout.addWidget(self.answer_display)
        body_layout.addWidget(answer_card, stretch=5)

        transcript_header = QHBoxLayout()
        transcript_title = QLabel("LIVE TRANSCRIPT")
        transcript_title.setObjectName("eyebrow")
        self.screen_meta = QLabel("SCREEN WATCH OFF")
        self.screen_meta.setObjectName("muted")
        transcript_header.addWidget(transcript_title)
        transcript_header.addStretch()
        transcript_header.addWidget(self.screen_meta)
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
        self.transcript_display.setText("Click Listen to start NVIDIA Parakeet transcription.")
        transcript_layout.addWidget(self.partial_transcript_label)
        transcript_layout.addWidget(self.transcript_display)

        transcript_actions = QHBoxLayout()
        transcript_actions.setContentsMargins(0, 0, 0, 0)
        transcript_actions.setSpacing(7)
        transcript_hint = QLabel("Send the current transcript to Gemini")
        transcript_hint.setObjectName("muted")
        self.transcript_send_btn = QPushButton("Send")
        self.transcript_send_btn.setObjectName("primaryButton")
        self.transcript_send_btn.setFixedWidth(76)
        self.transcript_send_btn.clicked.connect(self.trigger_text_analysis)
        transcript_actions.addWidget(transcript_hint)
        transcript_actions.addStretch()
        transcript_actions.addWidget(self.transcript_send_btn)
        transcript_layout.addLayout(transcript_actions)
        body_layout.addWidget(transcript_card, stretch=2)

        composer = QFrame()
        composer.setObjectName("composer")
        composer_row = QHBoxLayout(composer)
        composer_row.setContentsMargins(9, 7, 7, 7)
        composer_row.setSpacing(7)
        self.prompt_input = QLineEdit()
        self.prompt_input.setObjectName("promptInput")
        self.prompt_input.setPlaceholderText("Ask about the current conversation or screen…")
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
        font_size = max(11, int(self.settings.get("font_size", 13)))
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

    def moveEvent(self, event):
        super().moveEvent(event)
        if hasattr(self, "screen_watcher") and not self.settings.get("capture_region"):
            center = self.frameGeometry().center()
            self.screen_watcher.set_capture_target(point=(center.x(), center.y()))
            self._update_screen_exclusion()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "screen_watcher"):
            self._update_screen_exclusion()

    def _update_screen_exclusion(self):
        if not hasattr(self, "screen_watcher"):
            return
        geometry = self.frameGeometry()
        self.screen_watcher.set_exclude_rect(
            (geometry.left(), geometry.top(), geometry.width(), geometry.height())
        )

    def showEvent(self, event):
        super().showEvent(event)
        self.apply_invisible_mode()

    def apply_invisible_mode(self):
        try:
            set_window_invisible_to_capture(int(self.winId()), self.settings.get("invisible_mode", False))
        except Exception as exc:
            print(f"[overlay] Capture protection error: {exc}")

    def _sync_screen_watcher(self, restart=False):
        if not hasattr(self, "screen_watcher"):
            return
        enabled = bool(self.settings.get("auto_screen_watch", True) and _practice_mode_enabled())
        region = self.settings.get("capture_region")
        center = self.frameGeometry().center()
        self.screen_watcher.set_capture_target(
            region=region,
            point=None if region else (center.x(), center.y()),
        )
        self._update_screen_exclusion()
        if enabled and not self.screen_watcher.isRunning():
            self.screen_watcher.start()
        elif not enabled and self.screen_watcher.isRunning():
            self.screen_watcher.stop()
        elif restart and enabled and self.screen_watcher.isRunning():
            self.screen_watcher.stop()
            self.screen_watcher.interval_seconds = max(0.25, int(self.settings.get("screen_watch_interval_ms", 650)) / 1000.0)
            self.screen_watcher.stable_seconds = max(0.2, int(self.settings.get("screen_stable_ms", 450)) / 1000.0)
            self.screen_watcher.change_threshold = float(self.settings.get("screen_change_threshold", 0.055))
            self.screen_watcher.start()
        self.screen_meta.setText("SCREEN WATCH ON" if enabled else "SCREEN WATCH OFF")

    @Slot()
    def toggle_recording(self):
        if self.audio_recorder.is_recording:
            self.audio_recorder.stop_recording()
            self.record_btn.setText("Listen")
            self.partial_transcript_label.clear()
            self.partial_transcript_label.hide()
            self._set_status("IDLE")
            return

        if not _practice_mode_enabled():
            QMessageBox.information(
                self,
                "Practice mode is off",
                "Set PRACTICE_MODE=1 only for mock interviews or sessions where AI assistance is explicitly permitted.",
            )
            return
        if not self.get_effective_nvidia_key():
            QMessageBox.warning(
                self,
                "NVIDIA key missing",
                "Add NVIDIA_API_KEY to the project .env file and restart the app.",
            )
            return

        self.audio_recorder.set_devices(
            self.settings.get("mic_device_idx", -1),
            self.settings.get("system_device_idx", -1),
        )
        self.stt_worker.set_api_key(self.get_effective_nvidia_key())
        self.audio_recorder.start_recording()
        if not self.audio_recorder.is_recording:
            QMessageBox.warning(
                self,
                "Audio input unavailable",
                "No audio input could be opened. Choose a microphone or system loopback device in Settings.",
            )
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
            "Listening for speech…",
            "Listening starts automatically when practice mode is enabled.",
            "Click Listen to start NVIDIA Parakeet transcription.",
        }:
            self.transcript_display.clear()
        color = "#60A5FA" if speaker == "Candidate" else "#FBBF24"
        self.transcript_display.append(
            f'<b style="color:{color};">{speaker}</b><span style="color:#64748B;"> · </span>{text}<br>'
        )
        self.transcript_display.moveCursor(QTextCursor.End)

        if speaker != "Interviewer":
            return
        self.last_interviewer_time = time.monotonic()
        if not (
            self.settings.get("auto_answer_speech", True)
            and _practice_mode_enabled()
            and is_substantive_question(text)
        ):
            return

        cooldown = float(self.settings.get("answer_cooldown_seconds", 0.6))
        now = time.monotonic()
        if now - self.last_query_time < cooldown:
            return
        self.last_query_time = now

        image_bytes = None
        if self.settings.get("include_screen_with_speech", True) and self.latest_screen_bytes:
            max_age = float(self.settings.get("screen_context_max_age_seconds", 12.0))
            if now - self.latest_screen_time <= max_age and should_attach_screen(text):
                image_bytes = self.latest_screen_bytes

        self._enqueue_ai(
            source=f"Spoken question: *{text}*",
            kind="speech",
            image_bytes=image_bytes,
            custom_query=f"Answer this interviewer question now: {text}",
            use_image_history=False,
        )

    @Slot(bytes)
    def handle_screen_frame(self, image_bytes):
        self.latest_screen_bytes = image_bytes
        self.latest_screen_time = time.monotonic()
        self.screen_meta.setText("SCREEN CONTEXT LIVE")

        if self.audio_recorder.is_recording:
            return
        if not self.settings.get("auto_answer_screen", True):
            return
        now = time.monotonic()
        if now - self.last_screen_answer_time < 1.5:
            return
        self.last_screen_answer_time = now
        self._enqueue_ai(
            source="Screen changed",
            kind="screen",
            image_bytes=image_bytes,
            custom_query="Analyze the new stable screen and answer the visible practice question if there is one.",
            use_image_history=True,
        )

    def submit_screen_capture(self, image_bytes, source="Manual screen capture"):
        self.latest_screen_bytes = image_bytes
        self.latest_screen_time = time.monotonic()
        self._enqueue_ai(
            source=source,
            kind="manual_screen",
            image_bytes=image_bytes,
            custom_query="Solve or explain the visible practice question.",
            use_image_history=True,
        )

    @Slot(str)
    def update_status_log(self, status):
        print(f"[status] {status}")

    @Slot(str)
    def handle_stt_error(self, message):
        self.partial_transcript_label.hide()
        self._set_status("ERROR")
        self.answer_display.setMarkdown(f"### Voice transcription unavailable\n\n`{message}`")

    @Slot()
    def trigger_text_analysis(self):
        if not self.copilot_ai.transcript_history:
            self.answer_display.setMarkdown(
                "### Current transcript\n\nNo finalized transcript is available yet. Click Listen and wait for a spoken line to appear first."
            )
            return
        print(
            f"[transcript] Manual Send -> Gemini · lines={len(self.copilot_ai.transcript_history)}"
        )
        self._enqueue_ai(
            source="Current transcript",
            kind="chat",
            custom_query=(
                "Answer the latest substantive practice question from the live transcript now. "
                "Use the preceding transcript lines when the latest question depends on earlier context. "
                "For coding questions give the requested-language solution, explanation, complexity, and key edge cases. "
                "For MCQs put the correct option first with a concise reason."
            ),
        )

    @Slot()
    def send_custom_query(self):
        query = self.prompt_input.text().strip()
        if not query:
            return
        self.prompt_input.clear()
        image = None
        now = time.monotonic()
        if self.latest_screen_bytes and now - self.latest_screen_time <= float(
            self.settings.get("screen_context_max_age_seconds", 12.0)
        ):
            image = self.latest_screen_bytes if should_attach_screen(query) else None
        self._enqueue_ai(
            source=f"Chat: *{query}*",
            kind="chat",
            image_bytes=image,
            custom_query=query,
            use_image_history=False,
        )

    def _priority_for(self, kind):
        return {"speech": 0, "chat": 1, "manual_screen": 1, "screen": 3}.get(kind, 2)

    def _enqueue_ai(self, source, kind="chat", image_bytes=None, custom_query=None, use_image_history=False):
        self._configure_ai()
        if not _practice_mode_enabled():
            self.answer_display.setMarkdown(
                "### Practice mode is off\n\nSet `PRACTICE_MODE=1` only for permitted practice sessions."
            )
            return

        request_id = self.next_request_id
        self.next_request_id += 1
        request = {
            "id": request_id,
            "source": source,
            "kind": kind,
            "image_bytes": image_bytes,
            "custom_query": custom_query,
            "use_image_history": use_image_history,
            "priority": self._priority_for(kind),
            "created": time.monotonic(),
            "session": self.session_generation,
        }

        if kind in {"speech", "screen"}:
            self.request_queue = [item for item in self.request_queue if item["kind"] != kind]

        if kind == "speech":
            for active_id, active in list(self.requests.items()):
                if active.get("kind") == "screen":
                    self.stale_request_ids.add(active_id)

        self.request_queue.append(request)
        self.request_queue.sort(key=lambda item: (item["priority"], item["created"]))
        while len(self.request_queue) > 5:
            self.request_queue.pop()
        self._drain_ai_queue()

    def _drain_ai_queue(self):
        while self.request_queue and len(self.workers) < 2:
            if len(self.workers) == 1:
                only_active = next(iter(self.requests.values()))
                next_request = self.request_queue[0]
                if not (next_request["kind"] == "speech" and only_active.get("kind") == "screen"):
                    return

            request = self.request_queue.pop(0)
            request_id = request["id"]
            self.requests[request_id] = request
            self.streaming_text[request_id] = ""
            worker = AIQueryWorker(
                request_id,
                self.copilot_ai,
                image_bytes=request["image_bytes"],
                custom_query=request["custom_query"],
                use_image_history=request["use_image_history"],
            )
            self.workers[request_id] = worker
            worker.chunk_ready.connect(self._on_ai_chunk)
            worker.answer_ready.connect(self._on_ai_answer)
            worker.finished.connect(lambda rid=request_id: self._on_ai_worker_finished(rid))
            self._set_status("THINKING")
            if request_id not in self.stale_request_ids:
                self.answer_display.setMarkdown(f"### {request['source']}\n\n…")
            worker.start()

    @Slot(int, str)
    def _on_ai_chunk(self, request_id, chunk):
        request = self.requests.get(request_id)
        if not request or request_id in self.stale_request_ids or request.get("session") != self.session_generation:
            return
        self.streaming_text[request_id] = self.streaming_text.get(request_id, "") + chunk
        self.answer_display.setMarkdown(
            f"### {request['source']}\n\n{self.streaming_text[request_id]}"
        )
        self.answer_display.verticalScrollBar().setValue(0)

    @Slot(int, str)
    def _on_ai_answer(self, request_id, answer):
        request = self.requests.get(request_id)
        if not request or request_id in self.stale_request_ids or request.get("session") != self.session_generation:
            return
        answer = answer or self.streaming_text.get(request_id, "")
        block = f"### {request['source']}\n\n{answer}"
        self.answer_history.insert(0, block)
        self.answer_history = self.answer_history[:12]
        self.answer_display.setMarkdown("\n\n---\n\n".join(self.answer_history))
        self.answer_display.verticalScrollBar().setValue(0)

    def _on_ai_worker_finished(self, request_id):
        worker = self.workers.pop(request_id, None)
        self.requests.pop(request_id, None)
        self.streaming_text.pop(request_id, None)
        self.stale_request_ids.discard(request_id)
        if worker:
            worker.deleteLater()
        if not self.workers:
            self._set_status("LISTENING" if self.audio_recorder.is_recording else "IDLE")
        QTimer.singleShot(0, self._drain_ai_queue)

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
        self.session_generation += 1
        self.copilot_ai.clear_history()
        self.answer_history.clear()
        self.request_queue.clear()
        self.latest_screen_bytes = None
        for request_id, worker in list(self.workers.items()):
            self.stale_request_ids.add(request_id)
            worker.requestInterruption()
        self.transcript_display.setText("Transcript context cleared.")
        self.partial_transcript_label.hide()
        self.answer_display.setMarkdown("### Cleared\n\nReady for a new practice session.")

    @Slot()
    def open_settings(self):
        dialog = SettingsDialog(self.settings, self)
        if not dialog.exec():
            return

        was_recording = self.audio_recorder.is_recording
        old_devices = (
            self.settings.get("mic_device_idx", -1),
            self.settings.get("system_device_idx", -1),
        )
        if was_recording:
            self.audio_recorder.stop_recording()

        self.settings = dialog.settings.copy()
        self._configure_ai()
        self.mode_label.setText(self.copilot_ai.runtime_label())
        self.update_ui_stylesheet()
        self._apply_window_flags()
        self.show()
        self.apply_invisible_mode()
        self.setup_global_hotkeys()
        self.audio_recorder.set_devices(
            self.settings.get("mic_device_idx", -1),
            self.settings.get("system_device_idx", -1),
        )
        self.stt_worker.set_api_key(self.get_effective_nvidia_key())
        self._sync_screen_watcher(restart=True)

        new_devices = (
            self.settings.get("mic_device_idx", -1),
            self.settings.get("system_device_idx", -1),
        )
        if was_recording and _practice_mode_enabled() and self.get_effective_nvidia_key():
            self.audio_recorder.start_recording()
            if self.audio_recorder.is_recording:
                self.record_btn.setText("Stop")
                self._set_status("LISTENING")
            elif old_devices != new_devices:
                self.record_btn.setText("Listen")
                self._set_status("ERROR")
        elif (
            not was_recording
            and self.settings.get("auto_start_listening", True)
            and _practice_mode_enabled()
            and self.get_effective_nvidia_key()
        ):
            QTimer.singleShot(200, self.toggle_recording)

    def closeEvent(self, event):
        if self.hotkey_listener:
            self.hotkey_listener.stop()
        if hasattr(self, "screen_watcher"):
            self.screen_watcher.stop()
        self.audio_recorder.stop_recording()
        self.stt_worker.stop()
        controller = getattr(self, "mouse_passthrough_controller", None)
        if controller and hasattr(controller, "stop"):
            controller.stop()
        monitor = getattr(self, "audio_device_monitor", None)
        if monitor and hasattr(monitor, "stop"):
            monitor.stop()
        for worker in list(self.workers.values()):
            worker.requestInterruption()
            worker.wait(250)
        event.accept()
