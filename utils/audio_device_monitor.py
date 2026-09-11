import time

from PySide6.QtCore import QObject, QTimer

import config
from engine.audio_recorder import AudioRecorder, DEFAULT_SPEAKER_LOOPBACK_INDEX


class AudioDeviceMonitor(QObject):
    """Hot-plug monitor for microphones and system audio.

    New USB/Bluetooth microphones can be switched in while listening. On Windows,
    the synthetic ``-2`` system source follows the current default speaker through
    WASAPI loopback, so browser/video/meeting output keeps working when headphones
    or speakers change.
    """

    def __init__(self, window, interval_ms=1000):
        super().__init__(window)
        self.window = window
        self._last_mics, self._last_loopbacks = self._scan()
        self._last_refresh = 0.0
        self.timer = QTimer(self)
        self.timer.setInterval(max(650, int(interval_ms)))
        self.timer.timeout.connect(self._poll)
        self.timer.start()
        print("[audio-monitor] Hot-plug microphone/system-audio detection enabled.")

    @staticmethod
    def _scan():
        try:
            return AudioRecorder.list_devices()
        except Exception as exc:
            print(f"[audio-monitor] Device scan failed: {exc}")
            return [], []

    @staticmethod
    def _identity(item):
        return (
            str(item.get("name", "")).strip().lower(),
            str(item.get("api", "")).strip().lower(),
        )

    @staticmethod
    def _best_new_microphone(items):
        if not items:
            return None
        keywords = (
            "bluetooth",
            "headset",
            "hands-free",
            "hands free",
            "airpods",
            "buds",
            "jabra",
            "bose",
            "wh-",
            "wf-",
            "microphone",
        )
        ranked = sorted(
            items,
            key=lambda item: (
                -sum(
                    token in str(item.get("name", "")).lower()
                    for token in keywords
                ),
                -int(item.get("index", -1)),
            ),
        )
        return ranked[0]

    def stop(self):
        self.timer.stop()

    def _poll(self):
        if not self.window.settings.get("auto_detect_audio_devices", True):
            return

        mics, loopbacks = self._scan()
        old_mic_ids = {self._identity(item) for item in self._last_mics}
        old_loop_ids = {self._identity(item) for item in self._last_loopbacks}
        mic_ids = {self._identity(item) for item in mics}
        loop_ids = {self._identity(item) for item in loopbacks}
        new_mics = [item for item in mics if self._identity(item) not in old_mic_ids]
        new_loopbacks = [
            item for item in loopbacks if self._identity(item) not in old_loop_ids
        ]

        audio_topology_changed = mic_ids != old_mic_ids or loop_ids != old_loop_ids
        self._last_mics, self._last_loopbacks = mics, loopbacks
        if not audio_topology_changed:
            return

        self.refresh_active_devices(
            mics,
            loopbacks,
            new_mics,
            new_loopbacks,
            force_system_restart=(
                int(
                    self.window.settings.get("system_device_idx", -1)
                )
                == DEFAULT_SPEAKER_LOOPBACK_INDEX
                and loop_ids != old_loop_ids
            ),
        )

    def refresh_active_devices(
        self,
        mics=None,
        loopbacks=None,
        new_mics=None,
        new_loopbacks=None,
        force_system_restart=False,
    ):
        now = time.monotonic()
        if now - self._last_refresh < 0.65:
            return
        self._last_refresh = now

        if mics is None or loopbacks is None:
            mics, loopbacks = self._scan()
        new_mics = new_mics or []
        new_loopbacks = new_loopbacks or []

        current_mic = int(self.window.settings.get("mic_device_idx", -1))
        current_system = int(self.window.settings.get("system_device_idx", -1))
        valid_mic_indices = {int(item.get("index", -1)) for item in mics}
        valid_system_indices = {int(item.get("index", -1)) for item in loopbacks}

        selected_mic = current_mic
        selected_system = current_system

        if self.window.settings.get("auto_switch_new_microphone", True) and new_mics:
            preferred = self._best_new_microphone(new_mics)
            if preferred is not None:
                selected_mic = int(preferred.get("index", -1))
                print(
                    f"[audio-monitor] New microphone detected: "
                    f"{preferred.get('name')}"
                )
        elif current_mic not in valid_mic_indices:
            detected_mic, _ = AudioRecorder.auto_detect_devices()
            selected_mic = int(detected_mic)

        if current_system not in valid_system_indices:
            _, detected_system = AudioRecorder.auto_detect_devices()
            selected_system = int(detected_system)
        elif current_system < 0 and current_system != DEFAULT_SPEAKER_LOOPBACK_INDEX:
            if new_loopbacks:
                preferred_system = next(
                    (
                        item
                        for item in new_loopbacks
                        if int(item.get("index", -1))
                        == DEFAULT_SPEAKER_LOOPBACK_INDEX
                    ),
                    new_loopbacks[0],
                )
                selected_system = int(preferred_system.get("index", -1))

        changed_selection = (
            selected_mic != current_mic or selected_system != current_system
        )
        if not changed_selection and not force_system_restart:
            return

        was_recording = self.window.audio_recorder.is_recording
        if was_recording:
            self.window.audio_recorder.stop_recording()

        self.window.settings["mic_device_idx"] = selected_mic
        self.window.settings["system_device_idx"] = selected_system
        self.window.audio_recorder.set_devices(selected_mic, selected_system)
        config.save_settings(self.window.settings)

        if was_recording:
            self.window.audio_recorder.start_recording()
            if self.window.audio_recorder.is_recording:
                self.window.record_btn.setText("Stop")
                self.window._set_status("LISTENING")
                self.window.transcript_display.append(
                    "<span style='color:#86EFAC;'>"
                    "Audio routing changed — listening switched automatically."
                    "</span><br>"
                )
            else:
                self.window.record_btn.setText("Listen")
                self.window._set_status("ERROR")

        print(
            f"[audio-monitor] Active devices refreshed - "
            f"mic={selected_mic}, system={selected_system}."
        )
