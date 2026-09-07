import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

import config
from engine.audio_recorder import AudioRecorder
from ui.region_selector import RegionSelector

GEMINI_MODELS = ("gemini-2.5-flash", "gemini-2.5-pro")


class SettingsDialog(QDialog):
    def __init__(self, current_settings, parent=None):
        super().__init__(parent)
        self.settings = current_settings.copy()
        self.selector = None
        self.setWindowTitle("quntumnintent settings")
        self.setMinimumSize(620, 700)
        self.resize(680, 760)
        self._apply_style()
        self._build_ui()
        self._load_devices()

    def _apply_style(self):
        self.setStyleSheet("""
            QDialog { background:#070B12; color:#E7EEF8; font-family:'Segoe UI',Arial; font-size:13px; }
            QLabel { color:#C9D5E5; }
            QLabel#title { color:white; font-size:20px; font-weight:700; }
            QLabel#muted { color:#75859B; font-size:11px; }
            QFrame#card { background:#0C1420; border:1px solid #1F2C3E; border-radius:10px; }
            QComboBox, QSpinBox { min-height:34px; background:#111B2A; color:#F8FAFC; border:1px solid #2A3A50; border-radius:7px; padding:0 8px; }
            QCheckBox { color:#CAD5E3; spacing:8px; }
            QSlider::groove:horizontal { height:4px; background:#2A3A50; border-radius:2px; }
            QSlider::sub-page:horizontal { background:#3B82F6; }
            QSlider::handle:horizontal { width:16px; margin:-6px 0; background:#F8FAFC; border:2px solid #2563EB; border-radius:8px; }
            QPushButton { min-height:34px; border-radius:7px; padding:0 14px; font-weight:650; }
            QPushButton#primary { background:#2563EB; color:white; border:1px solid #3B82F6; }
            QPushButton#secondary { background:transparent; color:#B8C5D6; border:1px solid #334155; }
        """)

    def _card(self, title, subtitle=None):
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)
        heading = QLabel(title)
        heading.setStyleSheet("font-weight:700; color:#F8FAFC;")
        layout.addWidget(heading)
        if subtitle:
            text = QLabel(subtitle)
            text.setObjectName("muted")
            text.setWordWrap(True)
            layout.addWidget(text)
        return card, layout

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(12)

        title = QLabel("Assistant settings")
        title.setObjectName("title")
        root.addWidget(title)
        subtitle = QLabel("Credentials load from the project env file. Practice automation is enabled only when PRACTICE_MODE=1.")
        subtitle.setObjectName("muted")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        ai_card, ai_layout = self._card(
            "AI + live voice",
            "Gemini handles text/vision reasoning. NVIDIA Nemotron/Riva handles streaming speech-to-text.",
        )
        ai_form = QFormLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(GEMINI_MODELS)
        current_model = self.settings.get("model", config.DEFAULT_GEMINI_MODEL)
        idx = self.model_combo.findText(current_model)
        self.model_combo.setCurrentIndex(idx if idx >= 0 else 0)
        ai_form.addRow("Gemini model", self.model_combo)
        self.auto_start_check = QCheckBox("Start listening automatically")
        self.auto_start_check.setChecked(self.settings.get("auto_start_listening", True))
        self.auto_answer_check = QCheckBox("Answer substantive interviewer questions automatically")
        self.auto_answer_check.setChecked(self.settings.get("auto_answer_speech", True))
        ai_layout.addLayout(ai_form)
        ai_layout.addWidget(self.auto_start_check)
        ai_layout.addWidget(self.auto_answer_check)
        status = QLabel(
            f"Gemini: {'connected' if os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY') else 'key missing'}  ·  "
            f"NVIDIA: {'connected' if os.environ.get('NVIDIA_API_KEY') else 'key missing'}"
        )
        status.setObjectName("muted")
        ai_layout.addWidget(status)
        root.addWidget(ai_card)

        audio_card, audio_layout = self._card("Audio routing", "Choose the candidate microphone and an input-capable Windows loopback/system source.")
        audio_form = QFormLayout()
        self.mic_combo = QComboBox()
        self.system_combo = QComboBox()
        audio_form.addRow("Microphone", self.mic_combo)
        audio_form.addRow("System audio", self.system_combo)
        audio_layout.addLayout(audio_form)
        root.addWidget(audio_card)

        screen_card, screen_layout = self._card(
            "Real-time screen context",
            "The watcher captures locally and only emits a frame after a meaningful visual change becomes stable. While live listening is active it updates context without competing with the speech answer request.",
        )
        self.screen_watch_check = QCheckBox("Continuously watch the selected screen/region")
        self.screen_watch_check.setChecked(self.settings.get("auto_screen_watch", True))
        self.screen_answer_check = QCheckBox("Automatically answer stable screen-only questions when not listening")
        self.screen_answer_check.setChecked(self.settings.get("auto_answer_screen", True))
        self.include_screen_check = QCheckBox("Attach recent screen context when speech refers to visible code/error/question")
        self.include_screen_check.setChecked(self.settings.get("include_screen_with_speech", True))
        screen_layout.addWidget(self.screen_watch_check)
        screen_layout.addWidget(self.screen_answer_check)
        screen_layout.addWidget(self.include_screen_check)

        screen_form = QFormLayout()
        self.screen_interval_spin = QSpinBox()
        self.screen_interval_spin.setRange(300, 3000)
        self.screen_interval_spin.setSingleStep(50)
        self.screen_interval_spin.setSuffix(" ms")
        self.screen_interval_spin.setValue(int(self.settings.get("screen_watch_interval_ms", 650)))
        self.screen_stable_spin = QSpinBox()
        self.screen_stable_spin.setRange(200, 2000)
        self.screen_stable_spin.setSingleStep(50)
        self.screen_stable_spin.setSuffix(" ms")
        self.screen_stable_spin.setValue(int(self.settings.get("screen_stable_ms", 450)))
        screen_form.addRow("Capture interval", self.screen_interval_spin)
        screen_form.addRow("Stable before use", self.screen_stable_spin)
        screen_layout.addLayout(screen_form)

        region_row = QHBoxLayout()
        self.region_label = QLabel()
        self.region_label.setObjectName("muted")
        self._update_region_label()
        region_btn = QPushButton("Select region")
        region_btn.setObjectName("secondary")
        region_btn.clicked.connect(self._start_region_selection)
        clear_region_btn = QPushButton("Use monitor")
        clear_region_btn.setObjectName("secondary")
        clear_region_btn.clicked.connect(self._clear_region)
        region_row.addWidget(self.region_label, 1)
        region_row.addWidget(region_btn)
        region_row.addWidget(clear_region_btn)
        screen_layout.addLayout(region_row)
        root.addWidget(screen_card)

        overlay_card, overlay_layout = self._card(
            "Overlay",
            "Screen-capture exclusion is off by default. Enable it only when that behavior is appropriate for your permitted practice setup.",
        )
        self.invisible_check = QCheckBox("Exclude overlay from supported Windows capture APIs")
        self.invisible_check.setChecked(self.settings.get("invisible_mode", False))
        self.always_on_top_check = QCheckBox("Keep overlay always on top")
        self.always_on_top_check.setChecked(self.settings.get("always_on_top", True))
        overlay_layout.addWidget(self.invisible_check)
        overlay_layout.addWidget(self.always_on_top_check)

        overlay_form = QFormLayout()
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(55, 100)
        self.opacity_slider.setValue(int(float(self.settings.get("window_opacity", 0.94)) * 100))
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(11, 22)
        self.font_size_spin.setValue(int(self.settings.get("font_size", 13)))
        overlay_form.addRow("Opacity", self.opacity_slider)
        overlay_form.addRow("Answer font size", self.font_size_spin)
        overlay_layout.addLayout(overlay_form)
        root.addWidget(overlay_card)

        root.addStretch()
        footer = QHBoxLayout()
        footer.addStretch()
        cancel = QPushButton("Cancel")
        cancel.setObjectName("secondary")
        cancel.clicked.connect(self.reject)
        save = QPushButton("Save changes")
        save.setObjectName("primary")
        save.clicked.connect(self._save)
        footer.addWidget(cancel)
        footer.addWidget(save)
        root.addLayout(footer)

    def _load_devices(self):
        mics, loopbacks = AudioRecorder.list_devices()
        self.mic_combo.clear()
        self.system_combo.clear()
        self.mic_combo.addItem("Disabled", -1)
        self.system_combo.addItem("Disabled", -1)
        for item in mics:
            self.mic_combo.addItem(f"{item['name']} · {item.get('api', 'Audio')}", item["index"])
        for item in loopbacks:
            if item.get("index", -1) >= 0:
                self.system_combo.addItem(f"{item['name']} · {item.get('api', 'Loopback')}", item["index"])
        mic_idx = self.mic_combo.findData(self.settings.get("mic_device_idx", -1))
        sys_idx = self.system_combo.findData(self.settings.get("system_device_idx", -1))
        self.mic_combo.setCurrentIndex(mic_idx if mic_idx >= 0 else 0)
        self.system_combo.setCurrentIndex(sys_idx if sys_idx >= 0 else 0)

    def _update_region_label(self):
        region = self.settings.get("capture_region")
        if not region:
            self.region_label.setText("Watching the monitor containing the overlay")
        else:
            self.region_label.setText(
                f"Region {region.get('width', 0)}×{region.get('height', 0)} at ({region.get('left', 0)}, {region.get('top', 0)})"
            )

    def _start_region_selection(self):
        self.hide()
        self.selector = RegionSelector()
        self.selector.region_selected.connect(self._on_region_selected)
        self.selector.destroyed.connect(self._restore_after_selector)
        self.selector.show()
        self.selector.raise_()
        self.selector.activateWindow()

    def _restore_after_selector(self):
        if not self.isVisible():
            self.show()
            self.raise_()
            self.activateWindow()

    def _on_region_selected(self, top, left, width, height):
        self.settings["capture_region"] = {
            "top": int(top),
            "left": int(left),
            "width": int(width),
            "height": int(height),
        }
        self._update_region_label()
        self.show()
        self.raise_()
        self.activateWindow()

    def _clear_region(self):
        self.settings["capture_region"] = None
        self._update_region_label()

    def _save(self):
        self.settings["model"] = self.model_combo.currentText()
        self.settings["auto_start_listening"] = self.auto_start_check.isChecked()
        self.settings["auto_answer_speech"] = self.auto_answer_check.isChecked()
        self.settings["mic_device_idx"] = self.mic_combo.currentData()
        self.settings["system_device_idx"] = self.system_combo.currentData()
        self.settings["auto_screen_watch"] = self.screen_watch_check.isChecked()
        self.settings["auto_answer_screen"] = self.screen_answer_check.isChecked()
        self.settings["include_screen_with_speech"] = self.include_screen_check.isChecked()
        self.settings["screen_watch_interval_ms"] = self.screen_interval_spin.value()
        self.settings["screen_stable_ms"] = self.screen_stable_spin.value()
        self.settings["invisible_mode"] = self.invisible_check.isChecked()
        self.settings["always_on_top"] = self.always_on_top_check.isChecked()
        self.settings["window_opacity"] = self.opacity_slider.value() / 100.0
        self.settings["font_size"] = self.font_size_spin.value()
        config.save_settings(self.settings)
        self.accept()
