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
        self.setMinimumSize(620, 680)
        self.resize(680, 760)
        self._apply_style()
        self.init_ui()
        self.load_devices_and_populate()

    def _apply_style(self):
        self.setStyleSheet(
            """
            QDialog {
                background-color: #070B12;
                color: #E7EEF8;
                font-family: 'Segoe UI Variable Text', 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }
            QLabel { color: #C9D5E5; }
            QLabel#pageTitle { color: #FFFFFF; font-size: 21px; font-weight: 750; }
            QLabel#pageSubtitle { color: #7E8FA7; font-size: 11px; }
            QLabel#sectionTitle { color: #F7FAFC; font-size: 13px; font-weight: 700; }
            QLabel#sectionDescription, QLabel#fieldNote, QLabel#muted {
                color: #75859B; font-size: 11px;
            }
            QLabel#connected {
                color: #86EFAC;
                background: rgba(20,83,45,115);
                border: 1px solid #166534;
                border-radius: 9px;
                padding: 4px 9px;
                font-size: 10px;
                font-weight: 750;
            }
            QLabel#missing {
                color: #FCA5A5;
                background: rgba(127,29,29,105);
                border: 1px solid #991B1B;
                border-radius: 9px;
                padding: 4px 9px;
                font-size: 10px;
                font-weight: 750;
            }
            QFrame#sectionCard {
                background-color: #0C1420;
                border: 1px solid #1F2C3E;
                border-radius: 12px;
            }
            QComboBox, QSpinBox {
                min-height: 34px;
                background-color: #111B2A;
                color: #F8FAFC;
                border: 1px solid #2A3A50;
                border-radius: 7px;
                padding: 0 10px;
            }
            QComboBox:focus, QSpinBox:focus { border-color: #3B82F6; }
            QComboBox::drop-down { border: none; width: 28px; }
            QComboBox QAbstractItemView {
                background-color: #111B2A;
                color: #F8FAFC;
                border: 1px solid #2A3A50;
                selection-background-color: #1D4ED8;
            }
            QCheckBox { color: #CAD5E3; spacing: 8px; }
            QCheckBox::indicator {
                width: 16px; height: 16px;
                border: 1px solid #475569;
                border-radius: 4px;
                background: #111B2A;
            }
            QCheckBox::indicator:checked {
                background: #2563EB;
                border-color: #3B82F6;
            }
            QSlider::groove:horizontal { height: 4px; background: #2A3A50; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #3B82F6; border-radius: 2px; }
            QSlider::handle:horizontal {
                width: 16px; margin: -6px 0;
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
                background: #2563EB; color: white; border: 1px solid #3B82F6;
            }
            QPushButton#primaryButton:hover { background: #1D4ED8; }
            QPushButton#secondaryButton, QPushButton#regionButton {
                background: transparent; color: #B8C5D6; border: 1px solid #334155;
            }
            QPushButton#secondaryButton:hover, QPushButton#regionButton:hover {
                background: #172033; border-color: #475569; color: white;
            }
            QScrollArea { border: none; background: transparent; }
            QScrollArea > QWidget > QWidget { background: transparent; }
            QScrollBar:vertical { background: transparent; width: 8px; }
            QScrollBar::handle:vertical { background: #334155; border-radius: 4px; min-height: 28px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            """
        )

    def _section(self, title, description=None, status=None, status_ok=True):
        card = QFrame()
        card.setObjectName("sectionCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        header.addWidget(title_label)
        header.addStretch()
        if status:
            badge = QLabel(status)
            badge.setObjectName("connected" if status_ok else "missing")
            header.addWidget(badge)
        layout.addLayout(header)

        if description:
            text = QLabel(description)
            text.setObjectName("sectionDescription")
            text.setWordWrap(True)
            layout.addWidget(text)
        return card, layout

    @staticmethod
    def _add_field(grid, row, label_text, widget, note=None):
        label = QLabel(label_text)
        label.setMinimumWidth(128)
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
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(22, 20, 22, 14)
        top_layout.setSpacing(2)
        title = QLabel("Assistant settings")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Credentials load automatically from the project .env. Configure behavior here.")
        subtitle.setObjectName("pageSubtitle")
        top_layout.addWidget(title)
        top_layout.addWidget(subtitle)
        root.addWidget(top)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(22, 8, 22, 18)
        content_layout.setSpacing(12)

        gemini_ready = bool(
            os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        )
        ai_card, ai_layout = self._section(
            "AI reasoning",
            "Gemini handles chat, transcript answers, and screen/image understanding. The API key is read from .env and is never stored in app settings.",
            "CONNECTED" if gemini_ready else "KEY MISSING",
            gemini_ready,
        )
        ai_grid = QGridLayout()
        ai_grid.setHorizontalSpacing(14)
        ai_grid.setVerticalSpacing(8)
        ai_grid.setColumnStretch(1, 1)
        self.model_combo = QComboBox()
        self.model_combo.addItems(GEMINI_MODELS)
        current_model = self.settings.get("model", config.DEFAULT_GEMINI_MODEL)
        index = self.model_combo.findText(current_model)
        self.model_combo.setCurrentIndex(index if index >= 0 else 0)
        self._add_field(ai_grid, 0, "Gemini model", self.model_combo)
        ai_layout.addLayout(ai_grid)
        content_layout.addWidget(ai_card)

        nvidia_ready = bool(os.environ.get("NVIDIA_API_KEY", "").strip())
        voice_card, voice_layout = self._section(
            "Live voice",
            "NVIDIA Nemotron streams microphone and compatible Windows system audio into the transcript.",
            "CONNECTED" if nvidia_ready else "KEY MISSING",
            nvidia_ready,
        )
        self.auto_start_check = QCheckBox("Start listening automatically when the app opens")
        self.auto_start_check.setChecked(self.settings.get("auto_start_listening", True))
        self.auto_answer_check = QCheckBox("Answer detected interviewer questions automatically")
        self.auto_answer_check.setChecked(self.settings.get("auto_answer_speech", True))
        voice_layout.addWidget(self.auto_start_check)
        voice_layout.addWidget(self.auto_answer_check)
        content_layout.addWidget(voice_card)

        audio_card, audio_layout = self._section(
            "Audio routing",
            "Choose the candidate microphone and the loopback/system source used for remote speech.",
        )
        audio_grid = QGridLayout()
        audio_grid.setHorizontalSpacing(14)
        audio_grid.setVerticalSpacing(8)
        audio_grid.setColumnStretch(1, 1)
        self.mic_combo = QComboBox()
        self.system_combo = QComboBox()
        row = 0
        row = self._add_field(audio_grid, row, "Microphone", self.mic_combo)
        self._add_field(
            audio_grid,
            row,
            "System audio",
            self.system_combo,
            "Use a Windows loopback device to transcribe speech coming from a call or browser.",
        )
        audio_layout.addLayout(audio_grid)
        content_layout.addWidget(audio_card)

        capture_card, capture_layout = self._section(
            "Screen answer",
            "Capture the full screen or select a region. Gemini answers using both the image and recent transcript context.",
        )
        capture_row = QHBoxLayout()
        self.region_label = QLabel()
        self.region_label.setObjectName("muted")
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
            "Keep the assistant compact and readable while you practice.",
        )
        self.invisible_check = QCheckBox("Protect overlay from screen capture")
        self.invisible_check.setChecked(self.settings.get("invisible_mode", True))
        self.always_on_top_check = QCheckBox("Keep overlay always on top")
        self.always_on_top_check.setChecked(self.settings.get("always_on_top", True))
        overlay_layout.addWidget(self.invisible_check)
        overlay_layout.addWidget(self.always_on_top_check)

        overlay_grid = QGridLayout()
        overlay_grid.setHorizontalSpacing(14)
        overlay_grid.setVerticalSpacing(10)
        overlay_grid.setColumnStretch(1, 1)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(55)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(int(self.settings.get("window_opacity", 0.94) * 100))
        self.opacity_label = QLabel(f"{self.opacity_slider.value()}%")
        self.opacity_label.setObjectName("muted")
        self.opacity_slider.valueChanged.connect(lambda value: self.opacity_label.setText(f"{value}%"))
        opacity_widget = QWidget()
        opacity_row = QHBoxLayout(opacity_widget)
        opacity_row.setContentsMargins(0, 0, 0, 0)
        opacity_row.addWidget(self.opacity_slider, stretch=1)
        opacity_row.addWidget(self.opacity_label)

        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(11, 22)
        self.font_size_spin.setValue(self.settings.get("font_size", 13))
        row = 0
        row = self._add_field(overlay_grid, row, "Opacity", opacity_widget)
        self._add_field(overlay_grid, row, "Answer text", self.font_size_spin)
        overlay_layout.addLayout(overlay_grid)
        content_layout.addWidget(overlay_card)

        content_layout.addStretch()
        scroll.setWidget(content)
        root.addWidget(scroll, stretch=1)

        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(22, 14, 22, 18)
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
                f"{region.get('width', 0)} × {region.get('height', 0)} px · "
                f"x {region.get('left', 0)}, y {region.get('top', 0)}"
            )

    def load_devices_and_populate(self):
        mics, loopbacks = AudioRecorder.list_devices()
        self.mic_combo.clear()
        self.system_combo.clear()
        self.mic_combo.addItem("Disabled", -1)
        self.system_combo.addItem("Disabled", -1)

        for item in mics:
            self.mic_combo.addItem(f"{item['name']} · {item.get('api', 'Audio')}", item["index"])
        for item in loopbacks:
            self.system_combo.addItem(f"{item['name']} · {item.get('api', 'Loopback')}", item["index"])

        mic_index = self.mic_combo.findData(self.settings.get("mic_device_idx", -1))
        self.mic_combo.setCurrentIndex(mic_index if mic_index >= 0 else 0)
        system_index = self.system_combo.findData(self.settings.get("system_device_idx", -1))
        self.system_combo.setCurrentIndex(system_index if system_index >= 0 else 0)

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
        self.settings["auto_start_listening"] = self.auto_start_check.isChecked()
        self.settings["auto_answer_speech"] = self.auto_answer_check.isChecked()
        self.settings["mic_device_idx"] = self.mic_combo.currentData()
        self.settings["system_device_idx"] = self.system_combo.currentData()
        self.settings["invisible_mode"] = self.invisible_check.isChecked()
        self.settings["always_on_top"] = self.always_on_top_check.isChecked()
        self.settings["window_opacity"] = self.opacity_slider.value() / 100.0
        self.settings["font_size"] = self.font_size_spin.value()

        config.save_settings(self.settings)
        self.settings_saved.emit(self.settings)
        self.accept()
