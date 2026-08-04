import os
import sys
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QComboBox, QCheckBox, QPushButton, QSlider, QSpinBox, 
    QFormLayout, QGroupBox, QMessageBox, QWidget
)
from engine.audio_recorder import AudioRecorder
from ui.region_selector import RegionSelector
import config

class SettingsDialog(QDialog):
    settings_saved = Signal(dict)
    
    def __init__(self, current_settings, parent=None):
        super().__init__(parent)
        self.settings = current_settings.copy()
        
        self.setWindowTitle("quntumnintent - Settings")
        self.setMinimumWidth(480)
        self.resize(500, 600)
        
        # Apply modern dark-theme styling
        self.setStyleSheet("""
            QDialog {
                background-color: #1A1A1E;
                color: #E2E8F0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                color: #A0AEC0;
                font-size: 13px;
                font-weight: 500;
            }
            QGroupBox {
                border: 1px solid #2D3748;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
                font-weight: bold;
                color: #63B3ED;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                background-color: #1A1A1E;
            }
            QLineEdit, QComboBox, QSpinBox {
                background-color: #2D3748;
                border: 1px solid #4A5568;
                border-radius: 6px;
                padding: 6px 10px;
                color: #F7FAFC;
                font-size: 13px;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
                border: 1px solid #3182CE;
                background-color: #1E293B;
            }
            QPushButton {
                background-color: #3182CE;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #4299E1;
            }
            QPushButton:pressed {
                background-color: #2B6CB0;
            }
            QPushButton#cancelBtn {
                background-color: #4A5568;
                color: #E2E8F0;
            }
            QPushButton#cancelBtn:hover {
                background-color: #718096;
            }
            QPushButton#regionBtn {
                background-color: #2D3748;
                border: 1px dashed #63B3ED;
                color: #63B3ED;
            }
            QPushButton#regionBtn:hover {
                background-color: #3182CE;
                color: white;
            }
            QCheckBox {
                color: #E2E8F0;
                font-size: 13px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QSlider::handle:horizontal {
                background: #3182CE;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #4A5568;
            }
        """)
        
        self.init_ui()
        self.load_devices_and_populate()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # --- 1. AI API Settings Group ---
        ai_group = QGroupBox("AI Provider (.env credentials)")
        ai_layout = QFormLayout()
        ai_layout.setSpacing(10)
        
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["openai", "gemini", "ollama"])
        self.provider_combo.setCurrentText(self.settings.get("provider", "openai"))
        self.provider_combo.currentTextChanged.connect(self.on_provider_changed)
        
        self.model_combo = QComboBox()
        self.update_model_options()
        self.model_combo.setCurrentText(self.settings.get("model", "gpt-5.6-luna"))
        
        # Update dynamic states based on provider selection
        self.on_provider_changed(self.provider_combo.currentText())
        
        ai_layout.addRow("AI Provider:", self.provider_combo)
        ai_layout.addRow("Model Selection:", self.model_combo)
        ai_layout.addRow("Credentials:", QLabel("Loaded automatically from .env"))
        ai_group.setLayout(ai_layout)
        main_layout.addWidget(ai_group)
        
        # --- 2. Audio Settings Group ---
        audio_group = QGroupBox("Audio Sources Setup")
        audio_layout = QFormLayout()
        audio_layout.setSpacing(10)
        
        self.mic_combo = QComboBox()
        self.system_combo = QComboBox()

        self.stt_provider_combo = QComboBox()
        self.stt_provider_combo.addItems(["local", "gemini", "openai"])
        self.stt_provider_combo.setCurrentText(self.settings.get("stt_provider", "openai"))
        self.stt_provider_combo.currentTextChanged.connect(self.on_stt_provider_changed)

        self.stt_model_combo = QComboBox()
        self.stt_model_combo.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.stt_model_combo.setCurrentText(self.settings.get("stt_model", "base"))
        self.on_stt_provider_changed(self.stt_provider_combo.currentText())

        audio_layout.addRow("Your Voice (Mic):", self.mic_combo)
        audio_layout.addRow("Interviewer (System Output):", self.system_combo)
        audio_layout.addRow("STT Provider:", self.stt_provider_combo)
        audio_layout.addRow("Local Model:", self.stt_model_combo)
        audio_group.setLayout(audio_layout)
        main_layout.addWidget(audio_group)
        
        # --- 3. Screen Capture Region Group ---
        capture_group = QGroupBox("Screen Area Capture Settings")
        capture_layout = QVBoxLayout()
        capture_layout.setSpacing(10)
        
        self.region_label = QLabel()
        self.update_region_label()
        capture_layout.addWidget(self.region_label)
        
        self.select_region_btn = QPushButton("Select Screen Capture Region")
        self.select_region_btn.setObjectName("regionBtn")
        self.select_region_btn.clicked.connect(self.start_region_selection)
        capture_layout.addWidget(self.select_region_btn)
        
        capture_group.setLayout(capture_layout)
        main_layout.addWidget(capture_group)
        
        # --- 4. Interface Preferences Group ---
        ui_group = QGroupBox("Overlay Preferences")
        ui_layout = QFormLayout()
        ui_layout.setSpacing(10)
        
        # Invisible mode (The core requirement)
        self.invisible_check = QCheckBox("Invisible Mode (Protected Window)")
        self.invisible_check.setToolTip("Hides this overlay completely from screen sharing on Teams, Zoom, Slack, Meet, screenshots, and OBS.")
        self.invisible_check.setChecked(self.settings.get("invisible_mode", True))
        
        # Always on top
        self.always_on_top_check = QCheckBox("Always on Top")
        self.always_on_top_check.setChecked(self.settings.get("always_on_top", True))
        
        # Opacity slider
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(50)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(int(self.settings.get("window_opacity", 0.90) * 100))
        self.opacity_label = QLabel(f"{self.opacity_slider.value()}%")
        self.opacity_slider.valueChanged.connect(self.on_opacity_slider_changed)
        
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(self.opacity_slider)
        opacity_layout.addWidget(self.opacity_label)
        
        # Font size spinbox
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(10, 24)
        self.font_size_spin.setValue(self.settings.get("font_size", 13))
        
        ui_layout.addRow("Privacy Protect:", self.invisible_check)
        ui_layout.addRow("Keep On Top:", self.always_on_top_check)
        ui_layout.addRow("Overlay Opacity:", opacity_layout)
        ui_layout.addRow("Hint Text Size:", self.font_size_spin)
        ui_group.setLayout(ui_layout)
        main_layout.addWidget(ui_group)
        
        # --- 5. Action Buttons ---
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("cancelBtn")
        self.cancel_btn.clicked.connect(self.reject)
        
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.clicked.connect(self.save_and_accept)
        
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.save_btn)
        main_layout.addLayout(button_layout)

    def on_stt_provider_changed(self, text):
        # Disable local model dropdown unless using local Whisper
        if text == "local":
            self.stt_model_combo.setEnabled(True)
        else:
            self.stt_model_combo.setEnabled(False)

    def on_provider_changed(self, text):
        self.settings["provider"] = text
        self.update_model_options()
        
    def update_model_options(self):
        provider = self.provider_combo.currentText()
        self.model_combo.clear()
        
        if provider == "openai":
            self.model_combo.addItems(["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6"])
        elif provider == "gemini":
            self.model_combo.addItems(["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"])
        elif provider == "ollama":
            # Try to fetch local models dynamically from local Ollama tag server
            import requests
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=1.5)
                if response.status_code == 200:
                    models_data = response.json()
                    local_models = [m["name"] for m in models_data.get("models", [])]
                    if local_models:
                        self.model_combo.addItems(local_models)
                        return
            except Exception:
                pass
            
            # Fallback local models if Ollama API is not running or doesn't have downloaded models yet
            self.model_combo.addItems(["llama3", "mistral", "llava", "codellama", "phi3", "gemma2"])

    def on_opacity_slider_changed(self, value):
        self.opacity_label.setText(f"{value}%")

    def update_region_label(self):
        r = self.settings.get("capture_region")
        if r is None:
            self.region_label.setText("Active Region: [FULL SCREEN] (Auto-pilot captures entire primary monitor)")
        else:
            self.region_label.setText(
                f"Active Region: Box at ({r.get('left', 0)}, {r.get('top', 0)}) | Size: {r.get('width', 0)}x{r.get('height', 0)} px"
            )

    def load_devices_and_populate(self):
        """
        Dynamically queries the available audio inputs and WASAPI outputs (loopbacks)
        and populates the comboboxes.
        """
        mics, loopbacks = AudioRecorder.list_devices()
        
        self.mic_combo.clear()
        self.system_combo.clear()
        
        # Populate Microphones
        self.mic_devices_map = {}
        self.mic_combo.addItem("Disabled / None", -1)
        
        for m in mics:
            label = f"{m['name']} ({m.get('api', 'Unknown API')})"
            self.mic_combo.addItem(label, m['index'])
            self.mic_devices_map[m['index']] = label
            
        # Select current mic from settings
        target_mic = self.settings.get("mic_device_idx", -1)
        index = self.mic_combo.findData(target_mic)
        if index != -1:
            self.mic_combo.setCurrentIndex(index)
        else:
            self.mic_combo.setCurrentIndex(0) # Default to first (None)
            
        # Populate System Loopback
        self.system_devices_map = {}
        self.system_combo.addItem("Disabled / None", -1)
        
        for l in loopbacks:
            label = f"{l['name']} ({l.get('api', 'Unknown API')})"
            self.system_combo.addItem(label, l['index'])
            self.system_devices_map[l['index']] = label
            
        # Select current system audio from settings
        target_sys = self.settings.get("system_device_idx", -1)
        index = self.system_combo.findData(target_sys)
        if index != -1:
            self.system_combo.setCurrentIndex(index)
        else:
            # Try to auto-select loopback on windows if available
            auto_idx = -1
            for i in range(self.system_combo.count()):
                text = self.system_combo.itemText(i)
                if "loopback" in text.lower() or "[loopback]" in text.lower():
                    auto_idx = i
                    break
            if auto_idx != -1:
                self.system_combo.setCurrentIndex(auto_idx)
            else:
                self.system_combo.setCurrentIndex(0)

    def start_region_selection(self):
        """
        Temporarily hides the settings dialog and opens the transparent RegionSelector overlay.
        """
        self.hide()
        
        self.selector = RegionSelector()
        self.selector.region_selected.connect(self.on_region_selected)
        self.selector.show()
        
        # Ensure it stays on top and gains focus
        self.selector.activateWindow()
        self.selector.raise_()

    def on_region_selected(self, top, left, width, height):
        self.settings["capture_region"] = {
            "top": top,
            "left": left,
            "width": width,
            "height": height
        }
        self.update_region_label()
        
        # Reshow the settings dialog
        self.show()
        self.raise_()
        self.activateWindow()

    def save_and_accept(self):
        # Update settings dictionary from widgets
        self.settings["provider"] = self.provider_combo.currentText()
        self.settings["model"] = self.model_combo.currentText()
        self.settings["stt_provider"] = self.stt_provider_combo.currentText()
        self.settings["stt_model"] = self.stt_model_combo.currentText()
        
        self.settings["mic_device_idx"] = self.mic_combo.currentData()
        self.settings["system_device_idx"] = self.system_combo.currentData()
        
        self.settings["invisible_mode"] = self.invisible_check.isChecked()
        self.settings["always_on_top"] = self.always_on_top_check.isChecked()
        self.settings["window_opacity"] = self.opacity_slider.value() / 100.0
        self.settings["font_size"] = self.font_size_spin.value()
        
        # Save to disk
        config.save_settings(self.settings)
        
        # Emit signal and accept dialog
        self.settings_saved.emit(self.settings)
        self.accept()
