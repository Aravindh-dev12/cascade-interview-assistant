import os
import sys
import threading
from PySide6.QtCore import Qt, QPoint, Signal, Slot, QObject, QThread, QTimer
from PySide6.QtGui import QColor, QFont, QIcon, QTextCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextBrowser, 
    QLineEdit, QPushButton, QSizeGrip, QFrame, QMessageBox, QApplication
)
from pynput import keyboard

from engine.audio_recorder import AudioRecorder
from engine.stt_worker import STTWorker
from engine.screen_grabber import capture_screen, get_image_bytes
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
    
    def __init__(self, copilot_ai, image_bytes=None, custom_query=None):
        super().__init__()
        self.copilot_ai = copilot_ai
        self.image_bytes = image_bytes
        self.custom_query = custom_query

    def run(self):
        # Generate answer from AI
        answer = self.copilot_ai.generate_answer(
            image_bytes=self.image_bytes,
            custom_query=self.custom_query
        )
        self.finished.emit(answer)

class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        # Load user configurations
        self.settings = config.load_settings()
        
        # Get effective API keys with environment variable fallbacks
        stt_key = self.settings.get("api_key", "").strip() if self.settings.get("provider", "gemini") == "openai" else ""
        if not stt_key:
            stt_key = os.environ.get("OPENAI_API_KEY", "")
            
        copilot_key = self.get_effective_api_key()
        
        # Initialize Core Engines
        self.audio_recorder = AudioRecorder()
        self.stt_worker = STTWorker(self.audio_recorder, api_key=stt_key)
        self.copilot_ai = CopilotAI(
            provider=self.settings.get("provider", "gemini"),
            model=self.settings.get("model", "gemini-1.5-pro"),
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
        self.setMinimumSize(350, 400)
        self.resize(400, 650)
        
        # Draggable state
        self.drag_position = QPoint()
        
        # Hotkey listener setup (Only keep verbal toggle, remove screen analysis trigger)
        self.hotkey_signaler = HotkeySignaler()
        self.hotkey_signaler.record_hotkey_triggered.connect(self.toggle_recording)
        self.hotkey_listener = None
        self.setup_global_hotkeys()
        
        # Track the active trigger context (for "questions above, answers below" formatting)
        self.current_trigger_source = "Real-time automated capture"
        self.answer_history = []
        self.last_query_time = 0
        
        # Setup UI layout
        self.init_ui()
        
        # Connect transcription thread signals
        self.stt_worker.transcription_ready.connect(self.handle_transcription)
        self.stt_worker.status_updated.connect(self.update_status_log)
        self.stt_worker.error_occurred.connect(self.handle_stt_error)
        
        # Start transcription background worker
        self.stt_worker.start()

    def get_effective_api_key(self):
        """
        Gets the API key from settings, falling back to environment variables if empty.
        """
        settings_key = self.settings.get("api_key", "").strip()
        if settings_key:
            return settings_key
            
        provider = self.settings.get("provider", "openrouter")
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
        
        # Style sheet for translucent dark UI
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
        
        self.title_label = QLabel("🤖 quntumnintent")
        self.title_label.setStyleSheet("color: #E2E8F0; font-weight: bold; font-size: 14px;")
        
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        
        # Window buttons: Minimize, Close
        self.min_btn = QPushButton("—")
        self.min_btn.setObjectName("windowBtn")
        self.min_btn.clicked.connect(self.showMinimized)
        
        self.close_btn = QPushButton("✕")
        self.close_btn.setObjectName("closeBtn")
        self.close_btn.clicked.connect(self.close)
        
        title_layout.addWidget(self.min_btn)
        title_layout.addWidget(self.close_btn)
        
        # Install mouse drag filter on title bar
        title_bar.mousePressEvent = self.title_bar_mouse_press
        title_bar.mouseMoveEvent = self.title_bar_mouse_move
        
        container_layout.addWidget(title_bar)
        
        # Separator line
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #2D3748; max-height: 1px;")
        container_layout.addWidget(sep)
        
        # --- 2. AI Answer Display Panel (Markdown output) ---
        self.answer_display = QTextBrowser()
        self.answer_display.setOpenExternalLinks(True)
        self.answer_display.setMarkdown("### 🚀 Ready to Assist!\n\n"
                                        "Welcome to your quntumnintent.\n\n"
                                        "* **How to begin:** Configure your API keys and audio devices in Settings.\n"
                                        "* **Transcription:** Press **Start Listening** to capture system and mic audio.\n"
                                        "* **Vision Analysis:** Draw a capture region, then click **Capture Screen** or use **Ctrl+Shift+S** to solve code or diagrams instantly.\n\n"
                                        "*All interface windows are **protected and invisible** during screen shares (Teams, Zoom, Meet, etc.)!*")
        
        container_layout.addWidget(self.answer_display, stretch=4)
        
        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("background-color: #2D3748; max-height: 1px;")
        container_layout.addWidget(sep2)
        
        # --- 3. Live Transcripts Panel ---
        transcript_widget = QWidget()
        transcript_widget.setObjectName("transcriptPanel")
        transcript_layout = QVBoxLayout(transcript_widget)
        transcript_layout.setContentsMargins(10, 8, 10, 8)
        transcript_layout.setSpacing(4)
        
        trans_header = QHBoxLayout()
        trans_title = QLabel("🎙️ Live Interview Transcription")
        trans_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #A0AEC0;")
        self.status_led = QLabel("● IDLE")
        self.status_led.setStyleSheet("font-size: 10px; font-weight: bold; color: #718096;")
        trans_header.addWidget(trans_title)
        trans_header.addStretch()
        trans_header.addWidget(self.status_led)
        transcript_layout.addLayout(trans_header)
        
        self.transcript_display = QTextBrowser()
        self.transcript_display.setStyleSheet("""
            background-color: #0F172A; 
            border: 1px solid #1E293B; 
            border-radius: 4px; 
            color: #CBD5E1; 
            font-size: 11px;
        """)
        self.transcript_display.setText("Transcript history will appear here once audio recording starts...")
        transcript_layout.addWidget(self.transcript_display)
        
        container_layout.addWidget(transcript_widget, stretch=1)
        
        # --- 4. Custom Prompt / Command Entry ---
        prompt_bar = QWidget()
        prompt_bar.setStyleSheet("background-color: #1A1A1E; padding: 6px;")
        prompt_layout = QHBoxLayout(prompt_bar)
        prompt_layout.setContentsMargins(5, 0, 5, 0)
        prompt_layout.setSpacing(5)
        
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Ask custom query or type instructions here...")
        self.prompt_input.returnPressed.connect(self.send_custom_query)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.setStyleSheet("""
            background-color: #3182CE; 
            color: white; 
            border-radius: 4px; 
            padding: 6px 12px; 
            font-weight: bold;
        """)
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
        
        # Start/Stop Recording
        self.record_btn = QPushButton("🎤 Start Listening")
        self.record_btn.setObjectName("recordBtn")
        self.record_btn.clicked.connect(self.toggle_recording)
        
        # Clear History
        self.clear_btn = QPushButton("🗑️")
        self.clear_btn.setToolTip("Clear context transcripts")
        self.clear_btn.setStyleSheet("""
            QPushButton { 
                background-color: #2D3748; 
                font-size: 14px; 
                max-width: 34px; 
                min-width: 34px; 
                height: 34px; 
                border-radius: 4px; 
            }
            QPushButton:hover { background-color: #E53E3E; }
        """)
        self.clear_btn.clicked.connect(self.clear_context)
        
        # Size Grip for resizing borderless window
        size_grip = QSizeGrip(self)
        size_grip.setStyleSheet("width: 12px; height: 12px; image: none;") # Hides standard gray grips so it matches theme
        
        control_layout.addWidget(self.record_btn)
        control_layout.addWidget(self.clear_btn)
        control_layout.addWidget(size_grip, 0, Qt.AlignBottom | Qt.AlignRight)
        
        container_layout.addWidget(control_bar)
        
        root_layout.addWidget(self.container)

    def update_ui_stylesheet(self):
        opacity = self.settings.get("window_opacity", 0.90)
        font_size = self.settings.get("font_size", 13)
        
        # Dark color palette matching translucent overlay
        self.setStyleSheet(f"""
            QWidget {{
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
            QFrame#container {{
                background-color: rgba(26, 26, 30, {opacity});
                border: 1px solid #3182CE;
                border-radius: 8px;
            }}
            QWidget#titleBar {{
                background-color: rgba(15, 17, 26, 0.4);
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }}
            QPushButton#windowBtn {{
                background-color: transparent;
                border: none;
                color: #A0AEC0;
                font-size: 11px;
                max-width: 25px;
                height: 25px;
            }}
            QPushButton#windowBtn:hover {{
                color: white;
                background-color: #2D3748;
                border-radius: 3px;
            }}
            QPushButton#closeBtn {{
                background-color: transparent;
                border: none;
                color: #A0AEC0;
                font-size: 11px;
                max-width: 25px;
                height: 25px;
            }}
            QPushButton#closeBtn:hover {{
                color: white;
                background-color: #E53E3E;
                border-radius: 3px;
            }}
            QTextBrowser {{
                background-color: transparent;
                border: none;
                color: #E2E8F0;
                font-size: {font_size}px;
                line-height: 1.5;
                padding: 15px;
            }}
            /* Custom Slim Cyber-Blue Scrollbar for QTextBrowser */
            QTextBrowser QScrollBar:vertical {{
                border: none;
                background: rgba(30, 41, 59, 0.3);
                width: 8px;
                margin: 0px;
                border-radius: 4px;
            }}
            QTextBrowser QScrollBar::handle:vertical {{
                background: #3182CE;
                min-height: 25px;
                border-radius: 4px;
            }}
            QTextBrowser QScrollBar::handle:vertical:hover {{
                background: #4299E1;
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
                background-color: rgba(15, 17, 26, 0.4);
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
            }}
            QPushButton#recordBtn {{
                background-color: #E53E3E;
                color: white;
                border: none;
                border-radius: 4px;
                height: 34px;
                font-weight: bold;
                font-size: 12px;
                padding: 0 14px;
            }}
            QPushButton#recordBtn:hover {{
                background-color: #FC8181;
            }}
            QPushButton#captureBtn {{
                background-color: #059669;
                color: white;
                border: none;
                border-radius: 4px;
                height: 34px;
                font-weight: bold;
                font-size: 12px;
                padding: 0 14px;
            }}
            QPushButton#captureBtn:hover {{
                background-color: #34D399;
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
                    background-color: #E53E3E;
                }
                QPushButton#recordBtn:hover {
                    background-color: #FC8181;
                }
            """)
            self.status_led.setText("● IDLE")
            self.status_led.setStyleSheet("color: #718096; font-size: 10px; font-weight: bold;")
        else:
            # Start recording
            # Recheck api key in case it was updated in settings
            self.audio_recorder.set_devices(
                self.settings.get("mic_device_idx", -1),
                self.settings.get("system_device_idx", -1)
            )
            
            self.audio_recorder.start_recording()
            
            if not self.audio_recorder.mic_stream and not self.audio_recorder.system_stream:
                # If devices are not yet set up (defaulting to -1), don't raise a blocking error popup on launch.
                if self.settings.get("mic_device_idx", -1) == -1 and self.settings.get("system_device_idx", -1) == -1:
                    self.transcript_display.setText("[Real-Time Listening is Idle]\n\n"
                                                    "👉 Click the ⚙️ Gear Button below to choose your primary Microphone "
                                                    "and [Loopback] system speakers to start capturing call speech.")
                    self.audio_recorder.stop_recording()
                    self.record_btn.setText("🎤 Start Listening")
                    self.record_btn.setStyleSheet("""
                        QPushButton#recordBtn {
                            background-color: #E53E3E;
                        }
                        QPushButton#recordBtn:hover {
                            background-color: #FC8181;
                        }
                    """)
                    self.status_led.setText("● IDLE")
                    self.status_led.setStyleSheet("color: #718096; font-size: 10px; font-weight: bold;")
                    return
                else:
                    QMessageBox.critical(
                        self, 
                        "Audio Stream Error", 
                        "Failed to open any audio recording streams.\nPlease open Settings and verify your selected Mic and System devices."
                    )
                    self.audio_recorder.stop_recording()
                    return
                
            self.record_btn.setText("⏹️ Stop Listening")
            self.record_btn.setStyleSheet("""
                QPushButton#recordBtn {
                    background-color: #4A5568;
                }
                QPushButton#recordBtn:hover {
                    background-color: #718096;
                }
            """)
            self.status_led.setText("● LISTENING")
            self.status_led.setStyleSheet("color: #48BB78; font-size: 10px; font-weight: bold;")
            
            # Start / reset STT context
            self.stt_worker.set_api_key(self.settings.get("api_key", ""))
            
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
            
        color_code = "#4299E1" if speaker == "Candidate" else "#F6AD55"
        new_line = f'<b style="color: {color_code};">{speaker}:</b> {text}<br>'
        
        # Append as HTML to keep colors nice
        self.transcript_display.append(new_line)
        
        # Auto-scroll transcript window to bottom
        self.transcript_display.moveCursor(QTextCursor.End)

        # --- AUTO-PILOT AUTOMATIC TRIGGER FOR VERBAL SPEECH ---
        # If the Interviewer spoke a substantive phrase, automatically capture and answer!
        if speaker == "Interviewer" and len(text.strip()) > 8:
            import time
            current_time = time.time()
            if current_time - self.last_query_time > 4.0:  # 4 seconds minimum cooldown for verbal queries
                self.last_query_time = current_time
                print(f"[auto-pilot] Interviewer spoken question detected: \"{text}\" -> Triggering auto co-pilot...")
                self.current_trigger_source = f"🎙️ Spoken Interviewer Question:\n> *\"{text}\"*"
                self.trigger_text_analysis()

    def run_auto_vision_check(self):
        """
        Background tick that grabs the screen region and triggers analysis automatically
        if a significant visual shift is detected (new slide, scrolled text, new question).
        """
        # Only check if active recording is currently listening
        if not self.audio_recorder.is_recording:
            return
            
        region = self.settings.get("capture_region")
        try:
            current_img = capture_screen(region)
        except Exception:
            return # Silent catch in background loop
            
        # Compare with previous frame
        if self.last_captured_image is not None:
            if self.has_image_changed(self.last_captured_image, current_img, threshold=0.03):
                import time
                current_time = time.time()
                
                # Check 5.5-second cooldown for screen-update triggers to avoid API flooding
                if current_time - self.last_query_time > 5.5:
                    print("[auto-pilot] Screen update detected inside region! Triggering auto analysis...")
                    self.last_captured_image = current_img
                    self.last_query_time = current_time
                    self.current_trigger_source = "📸 Screen Visual Update (Automatic Full Monitor Grab)"
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
            self.settings.get("provider", "gemini"),
            self.settings.get("model", "gemini-2.5-flash"),
            self.get_effective_api_key()
        )
        
        self.ai_worker = AIQueryWorker(self.copilot_ai, custom_query=None)
        self.ai_worker.finished.connect(self.display_ai_answer)
        self.ai_worker.start()

    @Slot()
    def trigger_screen_analysis(self):
        """
        Grabs a screenshot of the selected active region and sends it 
        along with the speech transcript history to the Vision LLM for answers.
        """
        self.update_status_led_thinking(True)
        
        # Determine trigger source if not already set (fallback to manual capture)
        if not hasattr(self, "current_trigger_source") or not self.current_trigger_source:
            self.current_trigger_source = "📸 Screen Capture Trigger (Manual)"
            
        # Grab image of custom region
        region = self.settings.get("capture_region")
        print(f"[copilot] Capturing screen region: {region}")
        
        try:
            image = capture_screen(region)
            
            # --- LATENCY OPTIMIZATION ---
            # Downscale full screen images to keep them extremely light (max width 1280px)
            if image.width > 1280:
                scale_ratio = 1280.0 / float(image.width)
                new_height = int(float(image.height) * scale_ratio)
                from PIL import Image
                image = image.resize((1280, new_height), Image.Resampling.LANCZOS)
                
            # Compress heavily as high-efficiency JPEG instead of heavy PNG (drops size from 3MB to ~80KB!)
            image_bytes = get_image_bytes(image, format="JPEG", quality=60)
        except Exception as e:
            self.update_status_led_thinking(False)
            self.answer_display.setMarkdown(f"### ❌ Screen Capture Failed\n\nEnsure capture region is configured in settings.\n\nError: `{e}`")
            return

        self.answer_display.setMarkdown("### 🧠 Thinking...\n\nAnalyzing screen capture and interview transcript. Creating solution...")
        
        # Launch LLM request asynchronously inside QThread
        self.copilot_ai.set_config(
            self.settings.get("provider", "gemini"),
            self.settings.get("model", "gemini-2.5-flash"),
            self.get_effective_api_key()
        )
        
        self.ai_worker = AIQueryWorker(self.copilot_ai, image_bytes=image_bytes)
        self.ai_worker.finished.connect(self.display_ai_answer)
        self.ai_worker.start()

    @Slot()
    def send_custom_query(self):
        """
        Sends the typed text from the prompt box to the AI as a prioritized task.
        """
        query_text = self.prompt_input.text().strip()
        if not query_text:
            return
            
        self.prompt_input.clear()
        self.update_status_led_thinking(True)
        
        # Set trigger context
        self.current_trigger_source = f"💬 Typed Custom Query:\n> *\"{query_text}\"*"
        
        # Insert context into display
        self.answer_display.setMarkdown(f"### 🧠 Thinking...\n\nProcessing instruction: *\"{query_text}\"*...")
        
        # Launch LLM request asynchronously inside QThread
        self.copilot_ai.set_config(
            self.settings.get("provider", "gemini"),
            self.settings.get("model", "gemini-2.5-flash"),
            self.get_effective_api_key()
        )
        
        # Run query in background thread
        self.ai_worker = AIQueryWorker(self.copilot_ai, custom_query=query_text)
        self.ai_worker.finished.connect(self.display_ai_answer)
        self.ai_worker.start()

    @Slot(str)
    def display_ai_answer(self, markdown_text):
        """
        Renders markdown text answers received from the LLM background worker.
        """
        self.update_status_led_thinking(False)
        
        # Formulate current Q&A block: Questions on top, answers below!
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
            self.status_led.setStyleSheet("color: #3182CE; font-size: 10px; font-weight: bold;")
        else:
            if self.audio_recorder.is_recording:
                self.status_led.setText("● LISTENING")
                self.status_led.setStyleSheet("color: #48BB78; font-size: 10px; font-weight: bold;")
            else:
                self.status_led.setText("● IDLE")
                self.status_led.setStyleSheet("color: #718096; font-size: 10px; font-weight: bold;")

    @Slot()
    def clear_context(self):
        """
        Clears the local transcript display and the LLM transcript context.
        """
        self.copilot_ai.clear_history()
        self.answer_history = []  # Clear Q&A scrolling history list!
        self.transcript_display.clear()
        self.transcript_display.setText("[Context transcript history cleared]")
        self.answer_display.setMarkdown("### 🗑️ Transcript Context Cleared\n\nReady for new context.")

    @Slot()
    def open_settings(self):
        """
        Opens the settings dialog window, loading current values, 
        and updates configurations on safe close.
        """
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            # Retrieve updated configurations
            self.settings = dialog.settings.copy()
            
            # Reconfigure window properties dynamically
            self.update_ui_stylesheet()
            
            # Apply display protection dynamically
            self.apply_invisible_mode()
            
            # Set always-on-top dynamically
            is_on_top = self.settings.get("always_on_top", True)
            if is_on_top:
                self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
            else:
                self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
            self.show() # Showing is required after flag modification
            
            # Re-register global keyboard shortcuts in case they were updated
            self.setup_global_hotkeys()
            
            # Reconfigure engine options
            self.audio_recorder.set_devices(
                self.settings.get("mic_device_idx", -1),
                self.settings.get("system_device_idx", -1)
            )
            
            copilot_key = self.get_effective_api_key()
            self.copilot_ai.set_config(
                self.settings.get("provider", "gemini"),
                self.settings.get("model", "gemini-1.5-pro"),
                copilot_key
            )
            
            stt_key = self.settings.get("api_key", "").strip() if self.settings.get("provider", "gemini") == "openai" else ""
            if not stt_key:
                stt_key = os.environ.get("OPENAI_API_KEY", "")
            self.stt_worker.set_api_key(stt_key)
            
            print("[overlay] Settings successfully applied.")

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
            
        event.accept()
