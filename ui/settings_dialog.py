import os

from PySide6.QtCore import Qt
from PySide6.QtMultimedia import QMediaDevices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
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
from utils.camera_device_monitor import camera_device_id


class SettingsDialog(QDialog):
    def __init__(self, current_settings, parent=None):
        super().__init__(parent)
        self.settings = current_settings.copy()
        self.selector = None
        self.setWindowTitle("quntumnintent settings")
        self.setMinimumSize(620, 680)
        self.resize(700, 760)
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
            QScrollArea { border:none; background:transparent; }
            QScrollArea > QWidget > QWidget { background:transparent; }
            QToolTip { background:transparent; color:transparent; border:none; }
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
        root.setContentsMargins(0, 0, 0, 0)

        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(20, 18, 20, 10)
        title = QLabel("Assistant settings")
        title.setObjectName("title")
        subtitle = QLabel(
            "Google Gemini 2.5 Flash is the only answer engine. NVIDIA Parakeet handles speech-to-text and NVIDIA Nemotron handles first-pass screen/camera extraction."
        )
        subtitle.setObjectName("muted")
        subtitle.setWordWrap(True)
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        root.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(20, 8, 20, 12)
        content_layout.setSpacing(12)

        ai_card, ai_layout = self._card(
            "AI runtime",
            "Cloud-only runtime. No Ollama or local model is used. API keys stay in the project .env file and are never stored in Settings.",
        )
        ai_form = QFormLayout()
        self.provider_combo = QComboBox()
        self.provider_combo.addItem("Google Gemini 2.5 Flash only", "gemini")
        self.provider_combo.setEnabled(False)
        self.model_combo = QComboBox()
        self.model_combo.addItem(config.DEFAULT_GEMINI_MODEL, config.DEFAULT_GEMINI_MODEL)
        self.model_combo.setEnabled(False)
        ai_form.addRow("Answer engine", self.provider_combo)
        ai_form.addRow("Model", self.model_combo)
        ai_layout.addLayout(ai_form)

        nvidia_status = QLabel(
            "NVIDIA_API_KEY: loaded from .env"
            if os.environ.get("NVIDIA_API_KEY", "").strip()
            else "NVIDIA_API_KEY: missing from .env"
        )
        nvidia_status.setObjectName("muted")
        gemini_status = QLabel(
            "GEMINI_API_KEY: loaded from .env"
            if (
                os.environ.get("GEMINI_API_KEY", "").strip()
                or os.environ.get("GOOGLE_API_KEY", "").strip()
            )
            else "GEMINI_API_KEY: missing from .env"
        )
        gemini_status.setObjectName("muted")
        ai_layout.addWidget(nvidia_status)
        ai_layout.addWidget(gemini_status)
        content_layout.addWidget(ai_card)

        voice_card, voice_layout = self._card(
            "Live voice",
            "Listen captures microphone + system audio. NVIDIA Parakeet transcribes English speech; Gemini 2.5 Flash generates the final answer.",
        )
        self.auto_start_check = QCheckBox("Listening starts only when I click Listen")
        self.auto_start_check.setChecked(False)
        self.auto_start_check.setEnabled(False)
        self.auto_answer_check = QCheckBox("Answer substantive interviewer questions while listening")
        self.auto_answer_check.setChecked(self.settings.get("auto_answer_speech", True))
        self.auto_audio_check = QCheckBox("Auto-detect microphone and system-audio changes")
        self.auto_audio_check.setChecked(self.settings.get("auto_detect_audio_devices", True))
        self.auto_switch_mic_check = QCheckBox("Switch automatically to a newly connected microphone/headset")
        self.auto_switch_mic_check.setChecked(self.settings.get("auto_switch_new_microphone", True))
        voice_layout.addWidget(self.auto_start_check)
        voice_layout.addWidget(self.auto_answer_check)
        voice_layout.addWidget(self.auto_audio_check)
        voice_layout.addWidget(self.auto_switch_mic_check)

        audio_form = QFormLayout()
        self.mic_combo = QComboBox()
        self.system_combo = QComboBox()
        audio_form.addRow("Microphone", self.mic_combo)
        audio_form.addRow("System audio", self.system_combo)
        voice_layout.addLayout(audio_form)
        content_layout.addWidget(voice_card)

        camera_card, camera_layout = self._card(
            "Camera",
            "Camera analysis is manual. The latest frame is sent through NVIDIA visual extraction, then Gemini answers it; Gemini receives the image directly if NVIDIA vision is unavailable.",
        )
        camera_form = QFormLayout()
        self.camera_combo = QComboBox()
        camera_form.addRow("Camera", self.camera_combo)
        camera_layout.addLayout(camera_form)
        self.auto_camera_check = QCheckBox("Auto-detect camera connect/disconnect")
        self.auto_camera_check.setChecked(self.settings.get("auto_detect_camera_devices", True))
        self.auto_switch_camera_check = QCheckBox("Switch automatically to a newly connected camera")
        self.auto_switch_camera_check.setChecked(self.settings.get("auto_switch_new_camera", True))
        camera_layout.addWidget(self.auto_camera_check)
        camera_layout.addWidget(self.auto_switch_camera_check)
        content_layout.addWidget(camera_card)

        screen_card, screen_layout = self._card(
            "Manual screen capture",
            "No background screen inference. Click Capture screen or press Ctrl+Shift+S when you want one screenshot analyzed.",
        )
        self.screen_watch_check = QCheckBox("Background screen watching disabled")
        self.screen_watch_check.setChecked(False)
        self.screen_watch_check.setEnabled(False)
        self.screen_answer_check = QCheckBox("Automatic screen answering disabled")
        self.screen_answer_check.setChecked(False)
        self.screen_answer_check.setEnabled(False)
        self.include_screen_check = QCheckBox("Voice does not attach screenshots automatically")
        self.include_screen_check.setChecked(False)
        self.include_screen_check.setEnabled(False)
        screen_layout.addWidget(self.screen_watch_check)
        screen_layout.addWidget(self.screen_answer_check)
        screen_layout.addWidget(self.include_screen_check)

        self.screen_interval_spin = QSpinBox()
        self.screen_interval_spin.setRange(300, 3000)
        self.screen_interval_spin.setValue(int(self.settings.get("screen_watch_interval_ms", 650)))
        self.screen_interval_spin.hide()
        self.screen_stable_spin = QSpinBox()
        self.screen_stable_spin.setRange(200, 2000)
        self.screen_stable_spin.setValue(int(self.settings.get("screen_stable_ms", 450)))
        self.screen_stable_spin.hide()

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
        content_layout.addWidget(screen_card)

        overlay_card, overlay_layout = self._card("Overlay")
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
        content_layout.addWidget(overlay_card)
        content_layout.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        footer_widget = QWidget()
        footer = QHBoxLayout(footer_widget)
        footer.setContentsMargins(20, 10, 20, 16)
        footer.addStretch()
        cancel = QPushButton("Cancel")
        cancel.setObjectName("secondary")
        cancel.clicked.connect(self.reject)
        save = QPushButton("Save changes")
        save.setObjectName("primary")
        save.clicked.connect(self._save)
        footer.addWidget(cancel)
        footer.addWidget(save)
        root.addWidget(footer_widget)

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

        self.camera_combo.clear()
        self.camera_combo.addItem("No camera selected", "")
        try:
            for camera in QMediaDevices.videoInputs():
                cid = camera_device_id(camera)
                self.camera_combo.addItem(camera.description() or "Camera", cid)
        except Exception as exc:
            print(f"[settings] Camera enumeration failed: {exc}")
        camera_idx = self.camera_combo.findData(self.settings.get("camera_device_id", ""))
        self.camera_combo.setCurrentIndex(camera_idx if camera_idx >= 0 else 0)

    def _update_region_label(self):
        region = self.settings.get("capture_region")
        if not region:
            self.region_label.setText("Capture target: monitor")
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
        self.settings["ai_provider"] = "gemini"
        self.settings["model"] = config.DEFAULT_GEMINI_MODEL
        self.settings["auto_start_listening"] = False
        self.settings["auto_answer_speech"] = self.auto_answer_check.isChecked()
        self.settings["auto_detect_audio_devices"] = self.auto_audio_check.isChecked()
        self.settings["auto_switch_new_microphone"] = self.auto_switch_mic_check.isChecked()
        self.settings["mic_device_idx"] = self.mic_combo.currentData()
        self.settings["system_device_idx"] = self.system_combo.currentData()
        self.settings["camera_device_id"] = self.camera_combo.currentData() or ""
        self.settings["auto_detect_camera_devices"] = self.auto_camera_check.isChecked()
        self.settings["auto_switch_new_camera"] = self.auto_switch_camera_check.isChecked()
        self.settings["auto_screen_watch"] = False
        self.settings["auto_answer_screen"] = False
        self.settings["include_screen_with_speech"] = False
        self.settings["screen_watch_interval_ms"] = self.screen_interval_spin.value()
        self.settings["screen_stable_ms"] = self.screen_stable_spin.value()
        self.settings["invisible_mode"] = self.invisible_check.isChecked()
        self.settings["always_on_top"] = self.always_on_top_check.isChecked()
        self.settings["window_opacity"] = self.opacity_slider.value() / 100.0
        self.settings["font_size"] = self.font_size_spin.value()
        config.save_settings(self.settings)
        self.accept()
