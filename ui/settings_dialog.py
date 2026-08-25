import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

import config
from engine.audio_recorder import AudioRecorder
from ui.region_selector import RegionSelector


GEMINI_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.5-pro",
)


class SettingsDialog(QDialog):
    settings_saved = Signal(dict)

    def __init__(self, current_settings, parent=None):
        super().__init__(parent)
        self.settings = current_settings.copy()
        self.selector = None
        self.setWindowTitle("quntumnintent - Settings")
        self.setMinimumWidth(500)
        self.resize(520, 720)
        self._apply_style()
        self.init_ui()
        self.load_devices_and_populate()

    def _apply_style(self):
        self.setStyleSheet("""
            QDialog { background-color: #1A1A1E; color: #E2E8F0; font-family: 'Segoe UI', Arial, sans-serif; }
            QLabel { color: #A0AEC0; font-size: 13px; font-weight: 500; }
            QGroupBox { border: 1px solid #2D3748; border-radius: 8px; margin-top: 15px; padding-top: 15px; font-weight: bold; color: #63B3ED; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 8px; background-color: #1A1A1E; }
            QLineEdit, QComboBox, QSpinBox { background-color: #2D3748; border: 1px solid #4A5568; border-radius: 6px; padding: 6px 10px; color: #F7FAFC; font-size: 13px; }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border: 1px solid #3182CE; background-color: #1E293B; }
            QPushButton { background-color: #3182CE; color: white; border: none; border-radius: 6px; padding: 8px 16px; font-weight: bold; font-size: 13px; }
            QPushButton:hover { background-color: #4299E1; }
            QPushButton#cancelBtn { background-color: #4A5568; }
            QPushButton#regionBtn { background-color: #2D3748; border: 1px dashed #63B3ED; color: #63B3ED; }
            QCheckBox { color: #E2E8F0; font-size: 13px; }
            QSlider::handle:horizontal { background: #3182CE; width: 14px; margin: -5px 0; border-radius: 7px; }
            QSlider::groove:horizontal { height: 4px; background: #4A5568; }
        """)

    @staticmethod
    def _password_input(value, placeholder):
        field = QLineEdit()
        field.setEchoMode(QLineEdit.Password)
        field.setText(value or "")
        field.setPlaceholderText(placeholder)
        return field

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(12)

        gemini_group = QGroupBox("Gemini Answer Engine")
        gemini_layout = QFormLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(GEMINI_MODELS)
        current_model = self.settings.get("model", config.DEFAULT_GEMINI_MODEL)
        index = self.model_combo.findText(current_model)
        self.model_combo.setCurrentIndex(index if index >= 0 else 0)

        self.gemini_key_input = self._password_input(
            self.settings.get("api_key", ""),
            "Blank = use GEMINI_API_KEY from .env",
        )
        self.show_gemini_key = QCheckBox("Show Gemini API key")
        self.show_gemini_key.stateChanged.connect(
            lambda state: self.gemini_key_input.setEchoMode(
                QLineEdit.Normal if state == Qt.Checked.value else QLineEdit.Password
            )
        )

        gemini_layout.addRow("Provider:", QLabel("Google Gemini only"))
        gemini_layout.addRow("Model:", self.model_combo)
        gemini_layout.addRow("Gemini API key:", self.gemini_key_input)
        gemini_layout.addRow("", self.show_gemini_key)
        gemini_group.setLayout(gemini_layout)
        main_layout.addWidget(gemini_group)

        nvidia_group = QGroupBox("NVIDIA Real-Time Voice")
        nvidia_layout = QFormLayout()
        nvidia_layout.addRow("Model:", QLabel("nemotron-asr-streaming"))
        self.nvidia_key_input = self._password_input(
            self.settings.get("nvidia_api_key", ""),
            "Blank = use NVIDIA_API_KEY from .env",
        )
        self.show_nvidia_key = QCheckBox("Show NVIDIA API key")
        self.show_nvidia_key.stateChanged.connect(
            lambda state: self.nvidia_key_input.setEchoMode(
                QLineEdit.Normal if state == Qt.Checked.value else QLineEdit.Password
            )
        )
        nvidia_layout.addRow("NVIDIA API key:", self.nvidia_key_input)
        nvidia_layout.addRow("", self.show_nvidia_key)
        nvidia_group.setLayout(nvidia_layout)
        main_layout.addWidget(nvidia_group)

        audio_group = QGroupBox("Audio Sources")
        audio_layout = QFormLayout()
        self.mic_combo = QComboBox()
        self.system_combo = QComboBox()
        audio_layout.addRow("Microphone:", self.mic_combo)
        audio_layout.addRow("Laptop/System Audio:", self.system_combo)
        audio_group.setLayout(audio_layout)
        main_layout.addWidget(audio_group)

        capture_group = QGroupBox("Screen Capture Region")
        capture_layout = QVBoxLayout()
        self.region_label = QLabel()
        self.update_region_label()
        self.select_region_btn = QPushButton("Select Screen Capture Region")
        self.select_region_btn.setObjectName("regionBtn")
        self.select_region_btn.clicked.connect(self.start_region_selection)
        capture_layout.addWidget(self.region_label)
        capture_layout.addWidget(self.select_region_btn)
        capture_group.setLayout(capture_layout)
        main_layout.addWidget(capture_group)

        ui_group = QGroupBox("Overlay Preferences")
        ui_layout = QFormLayout()
        self.invisible_check = QCheckBox("Invisible Mode (Protected Window)")
        self.invisible_check.setChecked(self.settings.get("invisible_mode", True))
        self.always_on_top_check = QCheckBox("Always on Top")
        self.always_on_top_check.setChecked(self.settings.get("always_on_top", True))

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(50)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(int(self.settings.get("window_opacity", 0.90) * 100))
        self.opacity_label = QLabel(f"{self.opacity_slider.value()}%")
        self.opacity_slider.valueChanged.connect(
            lambda value: self.opacity_label.setText(f"{value}%")
        )
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(self.opacity_slider)
        opacity_row.addWidget(self.opacity_label)

        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(10, 24)
        self.font_size_spin.setValue(self.settings.get("font_size", 13))

        ui_layout.addRow("Privacy Protect:", self.invisible_check)
        ui_layout.addRow("Keep On Top:", self.always_on_top_check)
        ui_layout.addRow("Overlay Opacity:", opacity_row)
        ui_layout.addRow("Text Size:", self.font_size_spin)
        ui_group.setLayout(ui_layout)
        main_layout.addWidget(ui_group)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("cancelBtn")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save_and_accept)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(save_btn)
        main_layout.addLayout(buttons)

    def update_region_label(self):
        region = self.settings.get("capture_region")
        if region is None:
            self.region_label.setText("Active Region: Full Screen")
        else:
            self.region_label.setText(
                f"Active Region: ({region.get('left', 0)}, {region.get('top', 0)}) "
                f"{region.get('width', 0)}x{region.get('height', 0)}"
            )

    def load_devices_and_populate(self):
        mics, loopbacks = AudioRecorder.list_devices()
        self.mic_combo.clear()
        self.system_combo.clear()
        self.mic_combo.addItem("Disabled / None", -1)
        self.system_combo.addItem("Disabled / None", -1)

        for item in mics:
            self.mic_combo.addItem(
                f"{item['name']} ({item.get('api', 'Audio')})", item["index"]
            )
        for item in loopbacks:
            self.system_combo.addItem(
                f"{item['name']} ({item.get('api', 'Loopback')})", item["index"]
            )

        mic_index = self.mic_combo.findData(self.settings.get("mic_device_idx", -1))
        self.mic_combo.setCurrentIndex(mic_index if mic_index >= 0 else 0)
        sys_index = self.system_combo.findData(self.settings.get("system_device_idx", -1))
        self.system_combo.setCurrentIndex(sys_index if sys_index >= 0 else 0)

    def start_region_selection(self):
        self.hide()
        self.selector = RegionSelector()
        self.selector.region_selected.connect(self.on_region_selected)
        self.selector.show()
        self.selector.activateWindow()
        self.selector.raise_()

    def on_region_selected(self, top, left, width, height):
        self.settings["capture_region"] = {
            "top": top,
            "left": left,
            "width": width,
            "height": height,
        }
        self.update_region_label()
        self.show()
        self.raise_()
        self.activateWindow()

    def save_and_accept(self):
        self.settings["model"] = self.model_combo.currentText()
        self.settings["api_key"] = self.gemini_key_input.text().strip()
        self.settings["nvidia_api_key"] = self.nvidia_key_input.text().strip()
        self.settings["mic_device_idx"] = self.mic_combo.currentData()
        self.settings["system_device_idx"] = self.system_combo.currentData()
        self.settings["invisible_mode"] = self.invisible_check.isChecked()
        self.settings["always_on_top"] = self.always_on_top_check.isChecked()
        self.settings["window_opacity"] = self.opacity_slider.value() / 100.0
        self.settings["font_size"] = self.font_size_spin.value()

        missing = []
        if not self.settings["api_key"] and not (
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        ):
            missing.append("Gemini API key")
        if not self.settings["nvidia_api_key"] and not os.environ.get("NVIDIA_API_KEY"):
            missing.append("NVIDIA API key")
        if missing:
            QMessageBox.warning(
                self,
                "API Key Missing",
                "Missing: " + ", ".join(missing) + ". Add it here or in .env.",
            )

        config.save_settings(self.settings)
        self.settings_saved.emit(self.settings)
        self.accept()
