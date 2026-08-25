import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
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
        self.setWindowTitle("quntumnintent settings")
        self.setMinimumSize(620, 700)
        self.resize(660, 790)
        self._apply_style()
        self.init_ui()
        self.load_devices_and_populate()

    def _apply_style(self):
        self.setStyleSheet(
            """
            QDialog {
                background-color: #080D17;
                color: #E6EDF7;
                font-family: 'Segoe UI Variable Text', 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }
            QLabel { color: #C8D3E1; }
            QLabel#pageTitle { color: #F8FAFC; font-size: 20px; font-weight: 750; }
            QLabel#pageSubtitle { color: #77879C; font-size: 11px; }
            QLabel#sectionTitle { color: #F1F5F9; font-size: 13px; font-weight: 700; }
            QLabel#sectionDescription,
            QLabel#fieldNote,
            QLabel#valueMuted { color: #718096; font-size: 11px; }
            QLabel#providerBadge {
                color: #BFDBFE;
                background-color: rgba(30, 64, 175, 120);
                border: 1px solid #1D4ED8;
                border-radius: 9px;
                padding: 4px 8px;
                font-size: 10px;
                font-weight: 700;
            }
            QFrame#sectionCard {
                background-color: #0E1625;
                border: 1px solid #223047;
                border-radius: 11px;
            }
            QLineEdit,
            QComboBox,
            QSpinBox {
                min-height: 34px;
                background-color: #111C2D;
                color: #F8FAFC;
                border: 1px solid #2B3A50;
                border-radius: 7px;
                padding: 0 10px;
                selection-background-color: #1D4ED8;
            }
            QLineEdit:focus,
            QComboBox:focus,
            QSpinBox:focus {
                border-color: #3B82F6;
                background-color: #132036;
            }
            QComboBox::drop-down { border: none; width: 28px; }
            QComboBox QAbstractItemView {
                background-color: #111C2D;
                color: #F8FAFC;
                border: 1px solid #2B3A50;
                selection-background-color: #1D4ED8;
                outline: none;
            }
            QCheckBox { color: #C8D3E1; spacing: 8px; }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #475569;
                border-radius: 4px;
                background-color: #111C2D;
            }
            QCheckBox::indicator:checked {
                background-color: #2563EB;
                border-color: #3B82F6;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #2B3A50;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #3B82F6;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                margin: -6px 0;
                background: #F8FAFC;
                border: 2px solid #2563EB;
                border-radius: 8px;
            }
            QPushButton {
                min-height: 34px;
                border-radius: 7px;
                padding: 0 14px;
                font-weight: 650;
            }
            QPushButton#primaryButton {
                background-color: #2563EB;
                color: white;
                border: 1px solid #3B82F6;
            }
            QPushButton#primaryButton:hover { background-color: #1D4ED8; }
            QPushButton#secondaryButton {
                background-color: transparent;
                color: #B7C3D3;
                border: 1px solid #334155;
            }
            QPushButton#secondaryButton:hover {
                background-color: #172033;
                border-color: #475569;
                color: white;
            }
            QPushButton#regionButton {
                background-color: #111C2D;
                color: #C7D2E2;
                border: 1px solid #334155;
            }
            QPushButton#regionButton:hover {
                background-color: #172033;
                border-color: #3B82F6;
            }
            QScrollArea { border: none; background: transparent; }
            QScrollArea > QWidget > QWidget { background: transparent; }
            QScrollBar:vertical { background: transparent; width: 8px; }
            QScrollBar::handle:vertical {
                background: #334155;
                border-radius: 4px;
                min-height: 28px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical { height: 0; }
            """
        )

    @staticmethod
    def _password_input(value, placeholder):
        field = QLineEdit()
        field.setEchoMode(QLineEdit.Password)
        field.setText(value or "")
        field.setPlaceholderText(placeholder)
        return field

    def _section(self, title, description=None, badge=None):
        card = QFrame()
        card.setObjectName("sectionCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        header.setSpacing(8)
        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        header.addWidget(title_label)
        if badge:
            badge_label = QLabel(badge)
            badge_label.setObjectName("providerBadge")
            header.addWidget(badge_label)
        header.addStretch()
        layout.addLayout(header)

        if description:
            description_label = QLabel(description)
            description_label.setObjectName("sectionDescription")
            description_label.setWordWrap(True)
            layout.addWidget(description_label)

        return card, layout

    @staticmethod
    def _add_field(grid, row, label_text, widget, note=None):
        label = QLabel(label_text)
        label.setMinimumWidth(126)
        grid.addWidget(label, row, 0, Qt.AlignTop | Qt.AlignLeft)
        grid.addWidget(widget, row, 1)
        if note:
            note_label = QLabel(note)
            note_label.setObjectName("fieldNote")
            note_label.setWordWrap(True)
            grid.addWidget(note_label, row + 1, 1)
            return row + 2
        return row + 1

    def init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(22, 20, 22, 14)
        top_layout.setSpacing(12)

        copy = QVBoxLayout()
        copy.setSpacing(2)
        title = QLabel("Settings")
        title.setObjectName("pageTitle")
        subtitle = QLabel(
            "Configure AI, live voice transcription, audio routing, and overlay behavior."
        )
        subtitle.setObjectName("pageSubtitle")
        copy.addWidget(title)
        copy.addWidget(subtitle)
        top_layout.addLayout(copy)
        top_layout.addStretch()
        root.addWidget(top)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(22, 8, 22, 18)
        content_layout.setSpacing(12)

        gemini_card, gemini_layout = self._section(
            "AI reasoning",
            "Gemini is the only reasoning provider used for text and screen analysis.",
            "GEMINI",
        )
        gemini_grid = QGridLayout()
        gemini_grid.setHorizontalSpacing(14)
        gemini_grid.setVerticalSpacing(8)
        gemini_grid.setColumnStretch(1, 1)

        self.model_combo = QComboBox()
        self.model_combo.addItems(GEMINI_MODELS)
        current_model = self.settings.get("model", config.DEFAULT_GEMINI_MODEL)
        index = self.model_combo.findText(current_model)
        self.model_combo.setCurrentIndex(index if index >= 0 else 0)

        self.gemini_key_input = self._password_input(
            self.settings.get("api_key", ""),
            "Use GEMINI_API_KEY from .env when blank",
        )
        self.show_gemini_key = QCheckBox("Show API key")
        self.show_gemini_key.stateChanged.connect(
            lambda state: self.gemini_key_input.setEchoMode(
                QLineEdit.Normal if state == Qt.Checked.value else QLineEdit.Password
            )
        )

        row = 0
        row = self._add_field(gemini_grid, row, "Model", self.model_combo)
        row = self._add_field(
            gemini_grid,
            row,
            "API key",
            self.gemini_key_input,
            "Stored in local app settings when entered here. Leave blank to use .env.",
        )
        gemini_grid.addWidget(self.show_gemini_key, row, 1)
        gemini_layout.addLayout(gemini_grid)
        content_layout.addWidget(gemini_card)

        nvidia_card, nvidia_layout = self._section(
            "Live voice transcription",
            "NVIDIA Nemotron streaming ASR handles low-latency speech recognition.",
            "NVIDIA",
        )
        nvidia_grid = QGridLayout()
        nvidia_grid.setHorizontalSpacing(14)
        nvidia_grid.setVerticalSpacing(8)
        nvidia_grid.setColumnStretch(1, 1)

        model_value = QLabel("nemotron-asr-streaming")
        model_value.setObjectName("valueMuted")
        self.nvidia_key_input = self._password_input(
            self.settings.get("nvidia_api_key", ""),
            "Use NVIDIA_API_KEY from .env when blank",
        )
        self.show_nvidia_key = QCheckBox("Show API key")
        self.show_nvidia_key.stateChanged.connect(
            lambda state: self.nvidia_key_input.setEchoMode(
                QLineEdit.Normal if state == Qt.Checked.value else QLineEdit.Password
            )
        )

        row = 0
        row = self._add_field(nvidia_grid, row, "Model", model_value)
        row = self._add_field(
            nvidia_grid,
            row,
            "API key",
            self.nvidia_key_input,
            "Used only for NVIDIA Riva speech-to-text.",
        )
        nvidia_grid.addWidget(self.show_nvidia_key, row, 1)
        nvidia_layout.addLayout(nvidia_grid)
        content_layout.addWidget(nvidia_card)

        audio_card, audio_layout = self._section(
            "Audio routing",
            "Select the candidate microphone and system loopback source used for interviewer audio.",
        )
        audio_grid = QGridLayout()
        audio_grid.setHorizontalSpacing(14)
        audio_grid.setVerticalSpacing(8)
        audio_grid.setColumnStretch(1, 1)

        self.mic_combo = QComboBox()
        self.system_combo = QComboBox()
        row = 0
        row = self._add_field(audio_grid, row, "Microphone", self.mic_combo)
        row = self._add_field(
            audio_grid,
            row,
            "System audio",
            self.system_combo,
            "Windows loopback is required to capture call audio from the active output device.",
        )
        audio_layout.addLayout(audio_grid)
        content_layout.addWidget(audio_card)

        capture_card, capture_layout = self._section(
            "Screen capture",
            "Define the region Gemini should analyze when you trigger screen capture.",
        )
        capture_row = QHBoxLayout()
        capture_row.setSpacing(10)
        self.region_label = QLabel()
        self.region_label.setObjectName("valueMuted")
        self.update_region_label()
        self.select_region_btn = QPushButton("Select region")
        self.select_region_btn.setObjectName("regionButton")
        self.select_region_btn.clicked.connect(self.start_region_selection)
        capture_row.addWidget(self.region_label, stretch=1)
        capture_row.addWidget(self.select_region_btn)
        capture_layout.addLayout(capture_row)
        content_layout.addWidget(capture_card)

        overlay_card, overlay_layout = self._section(
            "Overlay",
            "Control visibility, window behavior, opacity, and reading size.",
        )
        overlay_grid = QGridLayout()
        overlay_grid.setHorizontalSpacing(14)
        overlay_grid.setVerticalSpacing(10)
        overlay_grid.setColumnStretch(1, 1)

        self.invisible_check = QCheckBox("Protect overlay from screen capture")
        self.invisible_check.setChecked(self.settings.get("invisible_mode", True))
        self.always_on_top_check = QCheckBox("Keep overlay always on top")
        self.always_on_top_check.setChecked(
            self.settings.get("always_on_top", True)
        )

        toggles = QWidget()
        toggles_layout = QVBoxLayout(toggles)
        toggles_layout.setContentsMargins(0, 0, 0, 0)
        toggles_layout.setSpacing(8)
        toggles_layout.addWidget(self.invisible_check)
        toggles_layout.addWidget(self.always_on_top_check)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(55)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(
            int(self.settings.get("window_opacity", 0.94) * 100)
        )
        self.opacity_label = QLabel(f"{self.opacity_slider.value()}%")
        self.opacity_label.setObjectName("valueMuted")
        self.opacity_label.setMinimumWidth(38)
        self.opacity_slider.valueChanged.connect(
            lambda value: self.opacity_label.setText(f"{value}%")
        )
        opacity_widget = QWidget()
        opacity_layout = QHBoxLayout(opacity_widget)
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        opacity_layout.setSpacing(10)
        opacity_layout.addWidget(self.opacity_slider, stretch=1)
        opacity_layout.addWidget(self.opacity_label)

        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(11, 22)
        self.font_size_spin.setValue(self.settings.get("font_size", 13))

        row = 0
        row = self._add_field(overlay_grid, row, "Behavior", toggles)
        row = self._add_field(overlay_grid, row, "Opacity", opacity_widget)
        row = self._add_field(overlay_grid, row, "Answer text size", self.font_size_spin)
        overlay_layout.addLayout(overlay_grid)
        content_layout.addWidget(overlay_card)

        content_layout.addStretch()
        scroll.setWidget(content)
        root.addWidget(scroll, stretch=1)

        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(22, 14, 22, 18)
        footer_layout.setSpacing(8)
        footer_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("secondaryButton")
        cancel_btn.clicked.connect(self.reject)

        save_btn = QPushButton("Save changes")
        save_btn.setObjectName("primaryButton")
        save_btn.clicked.connect(self.save_and_accept)

        footer_layout.addWidget(cancel_btn)
        footer_layout.addWidget(save_btn)
        root.addWidget(footer)

    def update_region_label(self):
        region = self.settings.get("capture_region")
        if region is None:
            self.region_label.setText("Full screen")
        else:
            self.region_label.setText(
                f"{region.get('width', 0)} × {region.get('height', 0)} px  ·  "
                f"x {region.get('left', 0)}, y {region.get('top', 0)}"
            )

    def load_devices_and_populate(self):
        mics, loopbacks = AudioRecorder.list_devices()
        self.mic_combo.clear()
        self.system_combo.clear()
        self.mic_combo.addItem("Disabled", -1)
        self.system_combo.addItem("Disabled", -1)

        for item in mics:
            self.mic_combo.addItem(
                f"{item['name']} · {item.get('api', 'Audio')}", item["index"]
            )
        for item in loopbacks:
            self.system_combo.addItem(
                f"{item['name']} · {item.get('api', 'Loopback')}", item["index"]
            )

        mic_index = self.mic_combo.findData(
            self.settings.get("mic_device_idx", -1)
        )
        self.mic_combo.setCurrentIndex(mic_index if mic_index >= 0 else 0)

        sys_index = self.system_combo.findData(
            self.settings.get("system_device_idx", -1)
        )
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
        if (
            not self.settings["nvidia_api_key"]
            and not os.environ.get("NVIDIA_API_KEY")
        ):
            missing.append("NVIDIA API key")

        if missing:
            QMessageBox.warning(
                self,
                "API configuration incomplete",
                "Missing: "
                + ", ".join(missing)
                + ". Add the key here or provide it through .env.",
            )

        config.save_settings(self.settings)
        self.settings_saved.emit(self.settings)
        self.accept()
