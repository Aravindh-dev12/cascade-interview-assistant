import time

from PySide6.QtCore import QObject, QTimer

import config
from engine.audio_recorder import AudioRecorder


class AudioDeviceMonitor(QObject):
    """Hot-plug monitor for microphones and input-capable system audio.

    When a new microphone appears (USB/Bluetooth/headset), the app can automatically
    switch to it and restart the active streams. If no new microphone appears, normal
    Windows default-device changes are still followed when the current selection
    disappears.
    """

    def __init__(self, window, interval_ms=1200):
        super().__init__(window)
        self.window = window
        self._last_mics, self._last_loopbacks = self._scan()
        self._last_refresh = 0.0
        self.timer = QTimer(self)
        self.timer.setInterval(max(700, int(interval_ms)))
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
        return str(item.get("name", "")).strip().lower(), str(item.get("api", "")).strip().lower()

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
                -sum(token in str(item.get("name", "")).lower() for token in keywords),
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
        new_mics = [item for item in mics if self._identity(item) not in old_mic_ids]
        new_loopbacks = [item for item in loopbacks if self._identity(item) not in old_loop_ids]

        changed = (
            {self._identity(item) for item in mics} != old_mic_ids
            or {self._identity(item) for item in loopbacks} != old_loop_ids
        )
        self._last_mics, self._last_loopbacks = mics, loopbacks
        if not changed:
            return

        self.refresh_active_devices(mics, loopbacks, new_mics, new_loopbacks)

    def refresh_active_devices(self, mics=None, loopbacks=None, new_mics=None, new_loopbacks=None):
        now = time.monotonic()
        if now - self._last_refresh < 0.8:
            return
        self._last_refresh = now

        mics = mics if mics is not None else self._scan()[0]
        loopbacks = loopbacks if loopbacks is not None else self._scan()[1]
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
                print(f"[audio-monitor] New microphone detected: {preferred.get('name')}")
        elif current_mic not in valid_mic_indices:
            detected_mic, _ = AudioRecorder.auto_detect_devices()
            selected_mic = int(detected_mic)

        if current_system not in valid_system_indices:
            _, detected_system = AudioRecorder.auto_detect_devices()
            selected_system = int(detected_system)
        elif current_system < 0 and new_loopbacks:
            selected_system = int(new_loopbacks[0].get("index", -1))

        if selected_mic == current_mic and selected_system == current_system:
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
                    "<span style='color:#86EFAC;'>Audio device changed — listening switched automatically.</span><br>"
                )
            else:
                self.window.record_btn.setText("Listen")
                self.window._set_status("ERROR")

        print(
            f"[audio-monitor] Active devices updated - mic={selected_mic}, system={selected_system}."
        )
