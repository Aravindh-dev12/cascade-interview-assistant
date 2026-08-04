import os
import sys
import threading
import re
from PySide6.QtCore import Qt, QPoint, Signal, Slot, QObject, QThread, QTimer
from PySide6.QtGui import QColor, QFont, QIcon, QTextCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextBrowser, 
    QLineEdit, QPushButton, QSizeGrip, QFrame, QMessageBox, QApplication,
    QComboBox
)
from pynput import keyboard

from engine.audio_recorder import AudioRecorder
from engine.stt_worker import STTWorker
from engine.screen_grabber import (
    capture_screen,
    get_image_bytes,
    images_are_near_duplicates,
    combine_scroll_captures,
)
from engine.copilot_ai import CopilotAI
from ui.settings_dialog import SettingsDialog
from utils.win_utils import set_window_invisible_to_capture
import config

# A helper QObject to bridge pynput global hotkeys to Qt signals
class HotkeySignaler(QObject):
    capture_hotkey_triggered = Signal()
    record_hotkey_triggered = Signal()

class AIQueryWorker(QThread):
    """
    Asynchronous QThread to run LLM requests (OpenAI/Anthropic/Gemini) 
    so the PySide6 UI event loop never freezes while waiting for network.
    """
    finished = Signal(str)
    partial = Signal(str)
    
    def __init__(
        self, copilot_ai, image_bytes=None, custom_query=None, spoken_only=False
    ):
        super().__init__()
        self.copilot_ai = copilot_ai
        self.image_bytes = image_bytes
        self.custom_query = custom_query
        self.spoken_only = spoken_only

    def run(self):
        # Generate answer from AI
        answer = self.copilot_ai.generate_answer(
            image_bytes=self.image_bytes,
            custom_query=self.custom_query,
            on_delta=self.partial.emit,
            spoken_only=self.spoken_only,
        )
        self.finished.emit(answer)

class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        # Load user configurations
        self.settings = config.load_settings()
        
        # Get effective API keys with environment variable fallbacks
        # For STT: use Gemini key by default (since default STT provider is now gemini)
        stt_provider = self.settings.get("stt_provider", "openai")
        stt_key = (
            os.environ.get("OPENAI_API_KEY", "")
            if stt_provider == "openai"
            else os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
        )
            
        copilot_key = self.get_effective_api_key()
        
        # Initialize Core Engines
        self.audio_recorder = AudioRecorder()
        self.stt_worker = STTWorker(
            self.audio_recorder,
            api_key=stt_key,
            stt_provider=self.settings.get("stt_provider", "openai"),
            stt_model=self.settings.get("stt_model", "base")
        )
        self.copilot_ai = CopilotAI(
            provider=self.settings.get("provider", "openai"),
            model=self.settings.get("model", "gpt-5.6-luna"),
            api_key=copilot_key
        )
        
        # Auto-detect your hardware: active microphone & loopback speakers on Windows!
        mic_idx, system_idx = AudioRecorder.auto_detect_devices()
        self.settings["mic_device_idx"] = mic_idx
        self.settings["system_device_idx"] = system_idx
        
        # Apply initial device configurations
        self.audio_recorder.set_devices(mic_idx, system_idx)
        
        # Frameless, translucent, always-on-top window setup (hidden from taskbar)
        self.setWindowFlags(
            Qt.FramelessWindowHint | 
            Qt.WindowStaysOnTopHint | 
            Qt.Tool if self.settings.get("always_on_top", True) else Qt.FramelessWindowHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMinimumSize(520, 500)
        self.resize(580, 780)
        
        # Draggable state
        self.drag_position = QPoint()
        
        # Hotkey listener setup
        self.hotkey_signaler = HotkeySignaler()
        self.hotkey_signaler.capture_hotkey_triggered.connect(self.trigger_screen_analysis)
        self.hotkey_signaler.record_hotkey_triggered.connect(self.toggle_recording)
        self.hotkey_listener = None
        self.setup_global_hotkeys()
        
        # Track the active trigger context (for "questions above, answers below" formatting)
        self.current_trigger_source = "Real-time automated capture"
        self.answer_history = []
        self.last_query_time = 0
        
        # Track running AI worker threads to prevent QThread destruction crash
        self.active_ai_workers = []
        
        # Track last captured screen image for auto-vision change detection
        self.last_captured_image = None
        self.scroll_captures = []
        self.last_scroll_capture_time = 0.0
        self.last_capture_region = None
        
        # Current selected code language
        self.current_language = "python"
        
        # Setup UI layout
        self.init_ui()
        self.apply_uniform_arrow_cursor()
        
        # Connect transcription thread signals
        self.stt_worker.transcription_ready.connect(self.handle_transcription)
        self.stt_worker.status_updated.connect(self.update_status_log)
        self.stt_worker.error_occurred.connect(self.handle_stt_error)
        
        # Start transcription background worker
        self.stt_worker.start()
        
        # Periodic audio level monitor so user can see if audio is flowing
        self.audio_monitor_timer = QTimer(self)
        self.audio_monitor_timer.timeout.connect(self._log_audio_levels)
        self.audio_monitor_timer.start(1000)  # Every 1 second
        
        # Auto-vision timer is DISABLED — screen capture is manual only (button or hotkey)
        # self.auto_vision_timer = QTimer(self)
        # self.auto_vision_timer.timeout.connect(self.run_auto_vision_check)
        # self.auto_vision_timer.start(1000)

    def get_effective_api_key(self):
        """
        Gets the selected provider's key only from environment variables.
        """
        provider = self.settings.get("provider", "openai")
        if provider == "openrouter":
            return os.environ.get("OPENROUTER_API_KEY", "")
        elif provider == "gemini":
            return os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
        elif provider == "openai":
            return os.environ.get("OPENAI_API_KEY", "")
        elif provider == "anthropic":
            return os.environ.get("ANTHROPIC_API_KEY", "")
        return ""

    def setup_global_hotkeys(self):
        """
        Launches the pynput global keyboard hotkey listener in a background thread.
        """
        try:
            # Stop existing listener if any
            if self.hotkey_listener:
                self.hotkey_listener.stop()
                
            hk_capture = self.settings.get("hotkey_capture", "<ctrl>+<shift>+s")
            hk_record = self.settings.get("hotkey_record", "<ctrl>+<shift>+a")
            
            # Map shortcut mappings to Qt-emitted triggers
            hotkey_map = {
                hk_capture: lambda: self.hotkey_signaler.capture_hotkey_triggered.emit(),
                hk_record: lambda: self.hotkey_signaler.record_hotkey_triggered.emit()
            }
            
            self.hotkey_listener = keyboard.GlobalHotKeys(hotkey_map)
            self.hotkey_listener.start()
            print(f"[hotkey] Hotkeys registered - Capture: {hk_capture}, Record: {hk_record}")
        except Exception as e:
            print(f"[hotkey] Error registering global hotkeys: {e}")

    def init_ui(self):
        # Root layout for overlay (simulates borders and shadow)
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        
        # The inner styled window card
        self.container = QFrame(self)
        self.container.setObjectName("container")
        self.container.setFrameShape(QFrame.StyledPanel)
        
        # Style sheet for light glassmorphism UI
        self.update_ui_stylesheet()
        
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        
        # --- 1. Custom Frameless Title Bar ---
        title_bar = QWidget()
        title_bar.setObjectName("titleBar")
        title_bar.setMinimumHeight(45)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(12, 0, 12, 0)
        
        self.title_label = QLabel("🤖 Interview Co-pilot")
        self.title_label.setStyleSheet("color: rgba(255, 255, 255, 0.95); font-weight: bold; font-size: 14px;")
        
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        
        # New Chat button
        self.new_chat_btn = QPushButton("➕")
        self.new_chat_btn.setObjectName("newChatBtn")
        self.new_chat_btn.setToolTip("Start a new chat session")
        self.new_chat_btn.clicked.connect(self.new_chat)
        
        # Window buttons: Minimize, Close
        self.min_btn = QPushButton("—")
        self.min_btn.setObjectName("windowBtn")
        self.min_btn.clicked.connect(self.showMinimized)
        
        self.close_btn = QPushButton("✕")
        self.close_btn.setObjectName("closeBtn")
        self.close_btn.clicked.connect(self.close)
        
        title_layout.addWidget(self.new_chat_btn)
        title_layout.addWidget(self.min_btn)
        title_layout.addWidget(self.close_btn)
        
        # Install mouse drag filter on title bar
        title_bar.mousePressEvent = self.title_bar_mouse_press
        title_bar.mouseMoveEvent = self.title_bar_mouse_move
        
        container_layout.addWidget(title_bar)
        
        # Separator line
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: rgba(255, 255, 255, 0.08); max-height: 1px;")
        container_layout.addWidget(sep)
        
        # --- 2. AI Answer Display Panel (Markdown output) ---
        self.answer_display = QTextBrowser()
        # Read-only chat content should keep the normal pointer, not show an
        # I-beam that makes it look like an editable typing area.
        self.answer_display.setCursor(Qt.ArrowCursor)
        self.answer_display.viewport().setCursor(Qt.ArrowCursor)
        self.answer_display.setOpenExternalLinks(True)
        self.answer_display.setMarkdown("### 🚀 Ready to Assist!\n\n"
                                        "Welcome to your Interview Co-pilot.\n\n"
                                        "* **How to begin:** Configure your API keys and audio devices in Settings.\n"
                                        "* **Transcription:** Click **Start Listening** to capture audio from your mic and system speakers.\n"
                                        "* **Screen Capture:** Click **📸 Capture Screen** (or **Ctrl+Shift+S**) to analyze any question on screen — coding, MCQ, theoretical, etc.\n"
                                        "* **Continuous Chat:** Type follow-up questions below — the AI remembers the full conversation context.\n\n"
                                        "*All interface windows are **protected and invisible** during screen shares (Teams, Zoom, Meet, etc.)!*")
        
        container_layout.addWidget(self.answer_display, stretch=4)
        
        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("background-color: rgba(255, 255, 255, 0.08); max-height: 1px;")
        container_layout.addWidget(sep2)
        
        # --- 3. Live Transcripts Panel ---
        transcript_widget = QWidget()
        transcript_widget.setObjectName("transcriptPanel")
        transcript_layout = QVBoxLayout(transcript_widget)
        transcript_layout.setContentsMargins(10, 8, 10, 8)
        transcript_layout.setSpacing(4)
        
        trans_header = QHBoxLayout()
        trans_title = QLabel("🎙️ Live Interview Transcription")
        trans_title.setStyleSheet("font-size: 11px; font-weight: bold; color: rgba(255, 255, 255, 0.6);")
        self.status_led = QLabel("● IDLE")
        self.status_led.setStyleSheet("font-size: 10px; font-weight: bold; color: rgba(255, 255, 255, 0.4);")
        trans_header.addWidget(trans_title)
        trans_header.addStretch()
        trans_header.addWidget(self.status_led)
        transcript_layout.addLayout(trans_header)
        
        self.transcript_display = QTextBrowser()
        self.transcript_display.setCursor(Qt.ArrowCursor)
        self.transcript_display.viewport().setCursor(Qt.ArrowCursor)
        self.transcript_display.setStyleSheet("""
            background-color: rgba(15, 23, 42, 0.25); 
            border: 1px solid rgba(255, 255, 255, 0.08); 
            border-radius: 6px; 
            color: rgba(255, 255, 255, 0.8); 
            font-size: 12px;
        """)
        self.transcript_display.setText("Transcript history will appear here once audio recording starts...")
        transcript_layout.addWidget(self.transcript_display)
        
        container_layout.addWidget(transcript_widget, stretch=2)

        # --- 4. Typed Follow-up Chat ---
        prompt_bar = QWidget()
        prompt_bar.setObjectName("promptBar")
        prompt_layout = QHBoxLayout(prompt_bar)
        prompt_layout.setContentsMargins(8, 6, 8, 6)
        prompt_layout.setSpacing(6)

        self.prompt_input = QLineEdit()
        # Keep the normal mouse pointer on hover; clicking still focuses the
        # field and allows normal keyboard typing/editing.
        self.prompt_input.setCursor(Qt.ArrowCursor)
        self.prompt_input.setPlaceholderText("")
        self.prompt_input.returnPressed.connect(self.send_custom_query)

        self.send_btn = QPushButton("➤")
        self.send_btn.setObjectName("sendBtn")
        self.send_btn.clicked.connect(self.send_custom_query)

        prompt_layout.addWidget(self.prompt_input)
        prompt_layout.addWidget(self.send_btn)
        container_layout.addWidget(prompt_bar)
        
        # --- 5. Controls Bottom Row Toolbar ---
        control_bar = QWidget()
        control_bar.setObjectName("controlBar")
        control_bar.setMinimumHeight(50)
        control_layout = QHBoxLayout(control_bar)
        control_layout.setContentsMargins(10, 5, 10, 5)
        control_layout.setSpacing(8)
        
        # Language selector dropdown
        self.lang_label = QLabel("Lang:")
        self.lang_label.setStyleSheet("color: rgba(255, 255, 255, 0.6); font-size: 11px; font-weight: bold;")
        self.lang_combo = QComboBox()
        self.lang_combo.setObjectName("langCombo")
        self.lang_combo.addItems(["Python", "C++", "C", "Java", "JavaScript", "Go", "Rust"])
        self.lang_combo.setCurrentText("Python")
        self.lang_combo.currentTextChanged.connect(self.on_language_changed)
        self.lang_combo.setFixedHeight(32)
        
        # Start/Stop Recording (audio only)
        self.record_btn = QPushButton("🎤 Start Listening")
        self.record_btn.setObjectName("recordBtn")
        self.record_btn.clicked.connect(self.toggle_recording)
        
        # Capture Screen button (manual screen capture)
        self.capture_btn = QPushButton("📸 Capture Screen")
        self.capture_btn.setObjectName("captureBtn")
        self.capture_btn.clicked.connect(self.trigger_screen_analysis)
        
        # Settings button
        self.settings_btn = QPushButton("⚙️")
        self.settings_btn.setObjectName("settingsBtn")
        self.settings_btn.setToolTip("Settings")
        self.settings_btn.clicked.connect(self.open_settings)
        
        # Size Grip for resizing borderless window
        size_grip = QSizeGrip(self)
        size_grip.setStyleSheet("width: 12px; height: 12px; image: none;")
        
        control_layout.addWidget(self.lang_label)
        control_layout.addWidget(self.lang_combo)
        control_layout.addWidget(self.record_btn)
        control_layout.addWidget(self.capture_btn)
        control_layout.addStretch()
        control_layout.addWidget(self.settings_btn)
        control_layout.addWidget(size_grip, 0, Qt.AlignBottom | Qt.AlignRight)
        
        container_layout.addWidget(control_bar)
        
        root_layout.addWidget(self.container)

    def update_ui_stylesheet(self):
        opacity = self.settings.get("window_opacity", 0.90)
        font_size = self.settings.get("font_size", 13)
        
        # True glassmorphism: dark frosted glass, high transparency, light text
        self.setStyleSheet(f"""
            QWidget {{
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
            QFrame#container {{
                background-color: rgba(15, 23, 42, {opacity * 0.55});
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 16px;
            }}
            QWidget#titleBar {{
                background-color: rgba(15, 23, 42, 0.15);
                border-top-left-radius: 16px;
                border-top-right-radius: 16px;
            }}
            QPushButton#windowBtn {{
                background-color: transparent;
                border: none;
                color: rgba(255, 255, 255, 0.6);
                font-size: 12px;
                max-width: 28px;
                height: 28px;
                border-radius: 6px;
            }}
            QPushButton#windowBtn:hover {{
                color: rgba(255, 255, 255, 0.95);
                background-color: rgba(255, 255, 255, 0.1);
            }}
            QPushButton#closeBtn {{
                background-color: transparent;
                border: none;
                color: rgba(255, 255, 255, 0.6);
                font-size: 12px;
                max-width: 28px;
                height: 28px;
                border-radius: 6px;
            }}
            QPushButton#closeBtn:hover {{
                color: white;
                background-color: #EF4444;
            }}
            QPushButton#newChatBtn {{
                background-color: transparent;
                border: none;
                color: rgba(255, 255, 255, 0.6);
                font-size: 14px;
                max-width: 28px;
                height: 28px;
                border-radius: 6px;
            }}
            QPushButton#newChatBtn:hover {{
                color: #60A5FA;
                background-color: rgba(96, 165, 250, 0.15);
            }}
            QTextBrowser {{
                background-color: transparent;
                border: none;
                color: rgba(255, 255, 255, 0.9);
                font-size: {font_size}px;
                line-height: 1.6;
                padding: 15px;
            }}
            QTextBrowser QScrollBar:vertical {{
                border: none;
                background: rgba(255, 255, 255, 0.05);
                width: 8px;
                margin: 0px;
                border-radius: 4px;
            }}
            QTextBrowser QScrollBar::handle:vertical {{
                background: rgba(255, 255, 255, 0.2);
                min-height: 25px;
                border-radius: 4px;
            }}
            QTextBrowser QScrollBar::handle:vertical:hover {{
                background: rgba(255, 255, 255, 0.35);
            }}
            QTextBrowser QScrollBar::add-line:vertical, QTextBrowser QScrollBar::sub-line:vertical {{
                border: none;
                background: none;
                height: 0px;
            }}
            QTextBrowser QScrollBar::up-arrow:vertical, QTextBrowser QScrollBar::down-arrow:vertical {{
                border: none;
                background: none;
            }}
            QWidget#controlBar {{
                background-color: rgba(15, 23, 42, 0.15);
                border-bottom-left-radius: 16px;
                border-bottom-right-radius: 16px;
            }}
            QWidget#promptBar {{
                background-color: transparent;
            }}
            QLineEdit {{
                background-color: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 8px;
                padding: 8px 12px;
                color: rgba(255, 255, 255, 0.9);
                font-size: 13px;
                selection-background-color: rgba(96, 165, 250, 0.3);
            }}
            QLineEdit::placeholder {{
                color: rgba(255, 255, 255, 0.35);
            }}
            QLineEdit:focus {{
                border: 1px solid rgba(96, 165, 250, 0.5);
                background-color: rgba(255, 255, 255, 0.12);
            }}
            QPushButton#sendBtn {{
                background-color: rgba(96, 165, 250, 0.8);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                max-width: 38px;
                min-width: 38px;
                height: 36px;
            }}
            QPushButton#sendBtn:hover {{
                background-color: rgba(96, 165, 250, 1.0);
            }}
            QPushButton#recordBtn {{
                background-color: rgba(239, 68, 68, 0.85);
                color: white;
                border: none;
                border-radius: 8px;
                height: 34px;
                font-weight: bold;
                font-size: 12px;
                padding: 0 16px;
            }}
            QPushButton#recordBtn:hover {{
                background-color: rgba(220, 38, 38, 0.95);
            }}
            QPushButton#captureBtn {{
                background-color: rgba(96, 165, 250, 0.8);
                color: white;
                border: none;
                border-radius: 8px;
                height: 34px;
                font-weight: bold;
                font-size: 12px;
                padding: 0 16px;
            }}
            QPushButton#captureBtn:hover {{
                background-color: rgba(96, 165, 250, 1.0);
            }}
            QPushButton#captureBtn:pressed {{
                background-color: rgba(59, 130, 246, 0.9);
            }}
            QPushButton#settingsBtn {{
                background-color: rgba(255, 255, 255, 0.08);
                color: rgba(255, 255, 255, 0.7);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                font-size: 16px;
                max-width: 38px;
                min-width: 38px;
                height: 34px;
            }}
            QPushButton#settingsBtn:hover {{
                background-color: rgba(255, 255, 255, 0.15);
            }}
            QComboBox#langCombo {{
                background-color: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 8px;
                padding: 4px 10px;
                color: rgba(255, 255, 255, 0.9);
                font-size: 12px;
                font-weight: bold;
                min-width: 80px;
            }}
            QComboBox#langCombo:hover {{
                border: 1px solid rgba(96, 165, 250, 0.4);
            }}
            QComboBox#langCombo::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox#langCombo::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid rgba(255, 255, 255, 0.5);
                width: 0;
                height: 0;
            }}
            QComboBox#langCombo QAbstractItemView {{
                background-color: rgba(15, 23, 42, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 6px;
                color: rgba(255, 255, 255, 0.9);
                padding: 4px;
                selection-background-color: rgba(96, 165, 250, 0.25);
            }}
        """)

    # --- Mouse Drag Math for Title Bar ---
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
        # Apply Windows Exclude from Capture Display Affinity
        self.apply_invisible_mode()

    def apply_invisible_mode(self):
        """
        Invokes native Windows API to exclude the window from screen sharing/recordings.
        """
        is_invisible = self.settings.get("invisible_mode", True)
        hwnd = int(self.winId())
        success = set_window_invisible_to_capture(hwnd, is_invisible)
        
        if is_invisible and success:
            print("[overlay] Native Windows capture protection engaged successfully.")
        else:
            print("[overlay] Capture protection inactive or failed.")

    # --- Core Event Handlers ---
    
    @Slot()
    def on_language_changed(self, lang_name):
        """Updates the preferred code language when the dropdown changes."""
        self.current_language = lang_name.lower().replace("++", "pp")
        self.copilot_ai.set_language(self.current_language)
        print(f"[copilot] Language changed to: {self.current_language}")
    
    @Slot()
    def new_chat(self):
        """Starts a fresh chat session - clears conversation history and display."""
        self.copilot_ai.clear_chat()
        self.answer_history = []
        self.scroll_captures.clear()
        self.last_scroll_capture_time = 0.0
        self.answer_display.setMarkdown("### ✨ New Chat Started\n\nReady for a new question. Capture a screen or type a question below.")
        print("[copilot] New chat session started - conversation history cleared.")
    
    @Slot()
    def toggle_recording(self):
        """
        Toggles state of the system audio loopback and mic recorders.
        """
        if self.audio_recorder.is_recording:
            # Stop recording
            self.audio_recorder.stop_recording()
            self.record_btn.setText("🎤 Start Listening")
            self.record_btn.setStyleSheet("""
                QPushButton#recordBtn {
                    background-color: rgba(239, 68, 68, 0.85);
                }
                QPushButton#recordBtn:hover {
                    background-color: rgba(220, 38, 38, 0.95);
                }
            """)
            self.status_led.setText("● IDLE")
            self.status_led.setStyleSheet("color: rgba(255, 255, 255, 0.4); font-size: 10px; font-weight: bold;")
        else:
            # Respect manually selected devices and auto-detect only missing ones.
            mic_idx = self.settings.get("mic_device_idx", -1)
            sys_idx = self.settings.get("system_device_idx", -1)
            if mic_idx is None or mic_idx < 0 or sys_idx is None or sys_idx < 0:
                print("[audio] Auto-detecting missing audio devices...")
                detected_mic, detected_sys = AudioRecorder.auto_detect_devices()
                if mic_idx is None or mic_idx < 0:
                    mic_idx = detected_mic
                if sys_idx is None or sys_idx < 0:
                    sys_idx = detected_sys
            
            # Save detected devices to settings for settings dialog display
            self.settings["mic_device_idx"] = mic_idx
            self.settings["system_device_idx"] = sys_idx
            print(f"[audio] Detected — Mic: {mic_idx}, System Loopback: {sys_idx}")
            
            self.audio_recorder.set_devices(mic_idx, sys_idx)
            self.audio_recorder.start_recording()
            
            # Check which streams actually started
            mic_ok = self.audio_recorder.mic_stream is not None
            sys_ok = self.audio_recorder.system_stream is not None
            
            if not mic_ok and not sys_ok:
                self.transcript_display.setText(
                    "⚠️ <b style='color:#FBBF24;'>No Audio Devices Found</b><br><br>"
                    "Could not detect any microphone or system audio device.<br>"
                    "Open ⚙️ Settings to manually select your devices.<br><br>"
                    "<b>Tip:</b> Make sure your earbuds/headphones are connected and selected as the default audio output in Windows."
                )
                self.audio_recorder.stop_recording()
                self.record_btn.setText("🎤 Start Listening")
                self.record_btn.setStyleSheet("""
                    QPushButton#recordBtn {
                        background-color: rgba(239, 68, 68, 0.85);
                    }
                    QPushButton#recordBtn:hover {
                        background-color: rgba(220, 38, 38, 0.95);
                    }
                """)
                self.status_led.setText("● IDLE")
                self.status_led.setStyleSheet("color: rgba(255, 255, 255, 0.4); font-size: 10px; font-weight: bold;")
                return
            elif not sys_ok:
                # Mic works but system loopback failed - CRITICAL for interview capture!
                self.transcript_display.setText(
                    "⚠️ <b style='color:#FBBF24;'>System Loopback Failed!</b><br>"
                    "Your mic is recording but interviewer audio is NOT being captured.<br><br>"
                    "<b>Fix:</b> Open ⚙️ Settings → choose a [Loopback] device for 'Interviewer (System Output)'.<br>"
                    "On Windows, pick your speakers/headphones under WASAPI."
                )
                print("[overlay] WARNING: Mic stream OK but system loopback stream FAILED. Interviewer audio will be blank.")
                
            self.record_btn.setText("⏹️ Stop Listening")
            self.record_btn.setStyleSheet("""
                QPushButton#recordBtn {
                    background-color: rgba(100, 116, 139, 0.7);
                }
                QPushButton#recordBtn:hover {
                    background-color: rgba(100, 116, 139, 0.85);
                }
            """)
            self.status_led.setText("● LISTENING")
            self.status_led.setStyleSheet("color: #10B981; font-size: 10px; font-weight: bold;")
            
            # Start / reset STT context — local whisper needs no API key
            stt_provider = self.settings.get("stt_provider", "openai")
            if stt_provider == "gemini":
                stt_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
            elif stt_provider == "openai":
                stt_key = os.environ.get("OPENAI_API_KEY", "")
            else:
                stt_key = ""
            self.stt_worker.set_api_key(stt_key)
            
            # Brief instruction on screen
            self.transcript_display.setText("[Listening starts... Audio is analyzed only when someone speaks]")

    @Slot(str, str)
    def handle_transcription(self, speaker, text):
        """
        Receives transcription updates from STTWorker thread, updates text history,
        and adds to the AI's conversation history buffer.
        """
        # Save to LLM context
        self.copilot_ai.add_transcript_line(speaker, text)
        
        # Display transcript in scrolling box
        current_transcript = self.transcript_display.toPlainText()
        if "will appear here once audio" in current_transcript or "Listening starts" in current_transcript:
            current_transcript = ""
            
        color_code = "#60A5FA" if speaker == "Candidate" else "#FBBF24"
        new_line = f'<b style="color: {color_code};">{speaker}:</b> {text}<br>'
        
        # Append as HTML to keep colors nice
        self.transcript_display.append(new_line)
        
        # Auto-scroll transcript window to bottom
        self.transcript_display.moveCursor(QTextCursor.End)

        # Trigger only for an actual question or interview assignment. Previously
        # every phrase ("yeah", "now you go", greetings) generated a nonsense answer.
        if speaker == "Interviewer" and self.is_interview_question(text):
            import time
            current_time = time.time()
            if current_time - self.last_query_time > 1.5:  # 1.5 sec cooldown for fast verbal response
                self.last_query_time = current_time
                print(f"[auto-pilot] Interviewer spoken question detected: \"{text}\" -> Triggering auto co-pilot...")
                self.current_trigger_source = f"🎙️ Spoken Interviewer Question:\n> *\"{text}\"*"
                self.answer_display.setMarkdown(
                    "### Heard the interviewer\n\n"
                    f"> {text}\n\n"
                    "Preparing a natural answer..."
                )
                self.trigger_text_analysis()

    @staticmethod
    def is_interview_question(text):
        """Return True for questions/tasks and False for conversational filler."""
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        # Interviewers often lead a real question with "yeah", "okay", or
        # "right, so". Remove those discourse markers before classification.
        normalized = re.sub(
            r"^(?:(?:yeah|yep|yes|okay|ok|right|great|good|sure)[,\s]+)+",
            "",
            normalized,
        )
        words = re.findall(r"[a-z0-9+#.-]+", normalized)
        if len(words) < 3:
            return False

        filler_patterns = (
            r"^(thanks|thank you)\b",
            r"^(now )?(you go|go ahead|your turn)\b",
            r"^(hello|hi|welcome|good morning|good afternoon)\b",
            r"^(can|could) you hear me\b",
            r"^(let'?s|we can|we will) (move|continue|start|begin)\b",
        )
        if any(re.search(pattern, normalized) for pattern in filler_patterns):
            return False

        interrogatives = {
            "what", "why", "how", "when", "where", "who", "which",
            "can", "could", "would", "will", "do", "does", "did",
            "is", "are", "was", "were", "should", "have", "has",
        }
        task_verbs = {
            "explain", "describe", "define", "compare", "differentiate",
            "discuss", "tell", "walk", "design", "implement", "write",
            "code", "solve", "find", "calculate", "compute", "prove",
            "optimize", "debug", "derive",
        }
        return (
            "?" in normalized
            or words[0] in interrogatives
            or words[0] in task_verbs
            or any(word in task_verbs for word in words[:4])
        )

    def run_auto_vision_check(self):
        """
        Background tick that grabs the screen and triggers analysis automatically
        if a significant visual shift is detected (new slide, new question, scrolled text, MCQ, etc.).
        Captures the configured region OR the full screen if no region is set.
        Works regardless of audio recording state so ALL screen questions are always captured.
        """
        # Use configured region, or fall back to full screen capture
        region = self.settings.get("capture_region")
        
        try:
            current_img = capture_screen(region)  # region=None captures full primary monitor
        except Exception:
            return  # Silent catch in background loop
            
        # Compare with previous frame
        if self.last_captured_image is not None:
            if self.has_image_changed(self.last_captured_image, current_img, threshold=0.02):
                import time
                current_time = time.time()
                
                # 3-second cooldown for screen-update triggers — fast response to new questions
                if current_time - self.last_query_time > 3.0:
                    print("[auto-pilot] Screen update detected! Triggering auto analysis...")
                    self.last_captured_image = current_img
                    self.last_query_time = current_time
                    self.current_trigger_source = "📸 Screen Visual Update (Auto Capture)"
                    self.trigger_screen_analysis()
            else:
                pass
        else:
            self.last_captured_image = current_img

    def has_image_changed(self, img1, img2, threshold=0.03):
        """
        Compares two frames using downsampled Grayscale MAE (Mean Absolute Error).
        Takes less than 1ms.
        """
        if img1 is None or img2 is None:
            return True
            
        try:
            # Resize and convert to L (grayscale)
            g1 = img1.resize((64, 64)).convert("L")
            g2 = img2.resize((64, 64)).convert("L")
            
            import numpy as np
            a1 = np.array(g1, dtype=np.float32) / 255.0
            a2 = np.array(g2, dtype=np.float32) / 255.0
            
            mae = np.mean(np.abs(a1 - a2))
            return mae > threshold
        except Exception as e:
            print(f"[auto-pilot] Error comparing images: {e}")
            return True

    @Slot(str)
    def update_status_log(self, status):
        print(f"[status] {status}")

    def _log_audio_levels(self):
        """Prints mic and system audio RMS levels to console and updates UI indicator."""
        if not self.audio_recorder.is_recording:
            return
        mic_rms = getattr(self.audio_recorder, 'mic_rms', 0)
        sys_rms = getattr(self.audio_recorder, 'system_rms', 0)
        mic_ok = "OK" if mic_rms > 0.001 else "SILENCE"
        sys_ok = "OK" if sys_rms > 0.001 else "SILENCE"
        print(f"[audio-levels] Mic: {mic_rms:.5f} ({mic_ok}) | System: {sys_rms:.5f} ({sys_ok})")
        
        # Update status LED with audio activity indicator
        if mic_rms > 0.001 or sys_rms > 0.001:
            self.status_led.setText("● LISTENING 🔊")
            self.status_led.setStyleSheet("color: #10B981; font-size: 10px; font-weight: bold;")
        else:
            self.status_led.setText("● LISTENING (quiet)")
            self.status_led.setStyleSheet("color: #FBBF24; font-size: 10px; font-weight: bold;")

    @Slot(str)
    def handle_stt_error(self, err_msg):
        # We don't want invasive messageboxes during active interviews,
        # so we display the error cleanly inside the display panel.
        self.answer_display.setMarkdown(f"### ⚠️ Speech-to-Text Error\n\n{err_msg}\n\n*Check your API connection or key.*")

    @Slot()
    def trigger_text_analysis(self):
        """
        Sends the transcript history to the LLM for text-only analysis (no screen capture).
        Used when interviewer speaks a question.
        """
        self.update_status_led_thinking(True)
        
        # Determine trigger source if not already set
        if not hasattr(self, "current_trigger_source") or not self.current_trigger_source:
            self.current_trigger_source = "🎙️ Audio Trigger (Verbal Question)"
            
        self.answer_display.setMarkdown("### 🧠 Thinking...\n\nAnalyzing interview transcript. Creating solution...")
        
        # Launch LLM request asynchronously inside QThread (text-only, no image)
        self.copilot_ai.set_config(
            self.settings.get("provider", "openai"),
            self.settings.get("model", "gpt-5.6-luna"),
            self.get_effective_api_key()
        )
        
        self._start_ai_worker(custom_query=None, spoken_only=True)

    @Slot()
    def trigger_screen_analysis(self):
        """
        Grabs a screenshot of the selected active region (or full screen) and sends it 
        along with the speech transcript history to the Vision LLM for answers.
        Handles ALL question types: coding, MCQ, theoretical, system design, SQL, etc.
        """
        self.update_status_led_thinking(True)
        
        # Determine trigger source if not already set (fallback to manual capture)
        if not hasattr(self, "current_trigger_source") or not self.current_trigger_source:
            self.current_trigger_source = "📸 Screen Capture (Manual)"
            
        # Grab image of custom region or full screen
        region = self.settings.get("capture_region")
        print(f"[copilot] Capturing screen region: {region if region else 'FULL SCREEN'}")
        
        try:
            image = capture_screen(region)

            # Captures made soon after scrolling are different views of the same
            # question. Keep up to three and send them together in reading order.
            import time
            now = time.time()
            region_signature = repr(region)
            if (
                now - self.last_scroll_capture_time > 12.0
                or region_signature != self.last_capture_region
            ):
                self.scroll_captures.clear()
            if (
                not self.scroll_captures
                or not images_are_near_duplicates(self.scroll_captures[-1], image)
            ):
                self.scroll_captures.append(image.copy())
                self.scroll_captures = self.scroll_captures[-3:]
            self.last_scroll_capture_time = now
            self.last_capture_region = region_signature
            image = combine_scroll_captures(self.scroll_captures)

            # Keep native pixels and lossless encoding. Downscaling/JPEG can alter
            # minus signs, decimal points, roots, subscripts and exponents.
            image_bytes = get_image_bytes(image, format="PNG")
        except Exception as e:
            self.update_status_led_thinking(False)
            self.answer_display.setMarkdown(f"### ❌ Screen Capture Failed\n\nError: `{e}`")
            return

        capture_count = len(self.scroll_captures)
        self.answer_display.setMarkdown(
            "### 🧠 Thinking...\n\n"
            f"Reconstructing the question from {capture_count} scroll view"
            f"{'s' if capture_count != 1 else ''}..."
        )
        
        # Launch LLM request asynchronously inside QThread
        self.copilot_ai.set_config(
            self.settings.get("provider", "openai"),
            self.settings.get("model", "gpt-5.6-luna"),
            self.get_effective_api_key()
        )
        
        self._start_ai_worker(image_bytes=image_bytes)

    @Slot()
    def send_custom_query(self):
        """Send a typed follow-up while preserving the current interview context."""
        query_text = self.prompt_input.text().strip()
        if not query_text:
            return

        self.prompt_input.clear()
        self.update_status_led_thinking(True)
        self.current_trigger_source = f"💬 Typed Follow-up:\n> *\"{query_text}\"*"
        self.answer_display.setMarkdown(
            f"### 🧠 Thinking...\n\nProcessing: *\"{query_text}\"*..."
        )
        self.copilot_ai.set_config(
            self.settings.get("provider", "openai"),
            self.settings.get("model", "gpt-5.6-luna"),
            self.get_effective_api_key(),
        )
        self._start_ai_worker(custom_query=query_text)

    def _start_ai_worker(
        self, image_bytes=None, custom_query=None, spoken_only=False
    ):
        """
        Creates, tracks, and starts an AIQueryWorker thread safely.
        Prevents QThread destruction crash by keeping references to all running workers.
        """
        worker = AIQueryWorker(
            self.copilot_ai,
            image_bytes=image_bytes,
            custom_query=custom_query,
            spoken_only=spoken_only,
        )
        worker.spoken_only = spoken_only
        worker.partial.connect(self.display_streaming_answer)
        worker.finished.connect(self.display_ai_answer)
        worker.finished.connect(lambda: self._cleanup_ai_worker(worker))
        self.active_ai_workers.append(worker)
        worker.start()

    @Slot(str)
    def display_streaming_answer(self, markdown_text):
        """Render partial model output immediately instead of waiting for completion."""
        self.answer_display.setMarkdown(markdown_text + "\n\n▌")
        self.answer_display.verticalScrollBar().setValue(0)

    def _cleanup_ai_worker(self, worker):
        """
        Removes a finished worker from the active list and safely quits it.
        """
        if worker in self.active_ai_workers:
            self.active_ai_workers.remove(worker)
        worker.quit()
        worker.wait(2000)

    @Slot(str)
    def display_ai_answer(self, markdown_text):
        """
        Renders markdown text answers received from the LLM background worker.
        """
        self.update_status_led_thinking(False)
        
        # Formulate current Q&A block: Questions on top, answers below!
        sender_worker = self.sender()
        spoken_only = bool(getattr(sender_worker, "spoken_only", False))
        if spoken_only:
            current_block = markdown_text.strip()
        else:
            trigger_context = f"### ❓ Context / Question\n{self.current_trigger_source}\n\n---\n\n"
            current_block = trigger_context + markdown_text
        
        # Prepend the newest answer to the history list (Newest on top!)
        self.answer_history.insert(0, current_block)
        
        # Limit history to the last 20 questions to preserve memory
        if len(self.answer_history) > 20:
            self.answer_history.pop()
            
        # Join past questions using standard Markdown horizontal divider lines (***)
        separator = "\n\n***\n\n"
        self.answer_display.setMarkdown(separator.join(self.answer_history))
        
        # Auto-scroll the answer space back to the TOP (0) so the candidate can read the latest answer immediately
        self.answer_display.verticalScrollBar().setValue(0)
        
        # Reset trigger source for next automated iteration
        self.current_trigger_source = None
        
        # Flash or raise overlay to notify user subtly (without grabbing OS window focus aggressively)
        self.activateWindow()

    def update_status_led_thinking(self, thinking):
        if thinking:
            self.status_led.setText("● THINKING...")
            self.status_led.setStyleSheet("color: #60A5FA; font-size: 10px; font-weight: bold;")
        else:
            if self.audio_recorder.is_recording:
                self.status_led.setText("● LISTENING")
                self.status_led.setStyleSheet("color: #10B981; font-size: 10px; font-weight: bold;")
            else:
                self.status_led.setText("● IDLE")
                self.status_led.setStyleSheet("color: rgba(255, 255, 255, 0.4); font-size: 10px; font-weight: bold;")

    @Slot()
    def clear_context(self):
        """
        Clears the local transcript display and the LLM transcript context.
        """
        self.copilot_ai.clear_history()
        self.transcript_display.clear()
        self.transcript_display.setText("[Transcript history cleared]")

    @Slot()
    def open_settings(self):
        """Open settings and apply the updated configuration."""
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            self.settings = dialog.settings.copy()
            self.update_ui_stylesheet()
            self.apply_uniform_arrow_cursor()
            self.apply_invisible_mode()

            is_on_top = self.settings.get("always_on_top", True)
            if is_on_top:
                self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
            else:
                self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
            self.show()
            self.setup_global_hotkeys()

            self.audio_recorder.set_devices(
                self.settings.get("mic_device_idx", -1),
                self.settings.get("system_device_idx", -1),
            )

            copilot_key = self.get_effective_api_key()
            self.copilot_ai.set_config(
                self.settings.get("provider", "openai"),
                self.settings.get("model", "gpt-5.6-luna"),
                copilot_key,
            )

            stt_provider = self.settings.get("stt_provider", "openai")
            if stt_provider == "gemini":
                stt_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
            elif stt_provider == "openai":
                stt_key = os.environ.get("OPENAI_API_KEY", "")
            else:
                stt_key = ""
            self.stt_worker.set_api_key(stt_key)
            self.stt_worker.set_stt_provider(
                self.settings.get("stt_provider", "openai"),
                self.settings.get("stt_model", "base"),
            )
            print("[overlay] Settings successfully applied.")

    def apply_uniform_arrow_cursor(self):
        """
        Keep the glass overlay as one interactive surface with no cursor or
        tooltip leakage from controls or the webpage/editor behind it.
        """
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setCursor(Qt.ArrowCursor)
        self.setToolTip("")
        for widget in self.findChildren(QWidget):
            widget.setCursor(Qt.ArrowCursor)
            widget.setToolTip("")
            if hasattr(widget, "viewport"):
                viewport = widget.viewport()
                if viewport is not None:
                    viewport.setCursor(Qt.ArrowCursor)

    def closeEvent(self, event):
        """
        Shut down active background threads safely on close.
        """
        print("[overlay] Shutting down application...")
        
        # Stop global hotkeys listener
        if self.hotkey_listener:
            self.hotkey_listener.stop()
            
        # Stop audio recording streams
        if self.audio_recorder:
            self.audio_recorder.stop_recording()
            
        # Stop STT background thread
        if self.stt_worker:
            self.stt_worker.stop()
            
        # Stop audio level monitor timer
        if hasattr(self, 'audio_monitor_timer'):
            self.audio_monitor_timer.stop()
        
        # Stop auto-vision timer (if enabled)
        if hasattr(self, 'auto_vision_timer') and self.auto_vision_timer:
            self.auto_vision_timer.stop()
        
        # Wait for all running AI worker threads to finish before closing
        for worker in self.active_ai_workers:
            worker.quit()
            worker.wait(3000)
        self.active_ai_workers.clear()
        
        event.accept()
