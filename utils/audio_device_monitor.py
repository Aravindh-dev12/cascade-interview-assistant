import time

from PySide6.QtCore import QObject, QTimer

from engine.audio_recorder import AudioRecorder


class AudioDeviceMonitor(QObject):
    """Keep active audio routing aligned with Windows device changes.

    This is intentionally conservative: it only restarts streams when the visible
    device topology changes while listening and automatic device detection is enabled.
    That covers common USB/Bluetooth connect/disconnect and Windows default-device
    changes without continuously disturbing a healthy ASR session.
    """

    def __init__(self, window, interval_ms=1500):
        super().__init__(window)
        self.window = window
        self._last_signature = self._device_signature()
        self._last_refresh = 0.0
        self.timer = QTimer(self)
        self.timer.setInterval(max(750, int(interval_ms)))
        self.timer.timeout.connect(self._poll)
        self.timer.start()
        print("[audio-monitor] Automatic Windows audio-device refresh enabled.")

    @staticmethod
    def _device_signature():
        try:
            mics, loopbacks = AudioRecorder.list_devices()
            mic_sig = tuple((item.get("index"), item.get("name")) for item in mics)
            loop_sig = tuple((item.get("index"), item.get("name")) for item in loopbacks)
            return mic_sig, loop_sig
        except Exception as exc:
            print(f"[audio-monitor] Device scan failed: {exc}")
            return (), ()

    def stop(self):
        self.timer.stop()

    def _poll(self):
        if not self.window.settings.get("auto_detect_audio_devices", True):
            return

        signature = self._device_signature()
        if signature == self._last_signature:
            return

        self._last_signature = signature
        print("[audio-monitor] Audio topology changed; refreshing default devices.")
        self.refresh_active_devices()

    def refresh_active_devices(self):
        """Re-detect defaults and hot-swap streams when listening is active."""
        now = time.monotonic()
        if now - self._last_refresh < 1.0:
            return
        self._last_refresh = now

        mic_idx, system_idx = AudioRecorder.auto_detect_devices()
        old_mic = self.window.settings.get("mic_device_idx", -1)
        old_system = self.window.settings.get("system_device_idx", -1)

        if mic_idx == old_mic and system_idx == old_system:
            return

        self.window.settings["mic_device_idx"] = mic_idx
        self.window.settings["system_device_idx"] = system_idx
        self.window.audio_recorder.set_devices(mic_idx, system_idx)

        if not self.window.audio_recorder.is_recording:
            print(
                f"[audio-monitor] Defaults updated - mic={mic_idx}, system={system_idx}."
            )
            return

        self.window.audio_recorder.stop_recording()
        self.window.audio_recorder.set_devices(mic_idx, system_idx)
        self.window.audio_recorder.start_recording()

        if self.window.audio_recorder.mic_stream or self.window.audio_recorder.system_stream:
            self.window.record_btn.setText("Stop")
            self.window._set_status("LISTENING")
            self.window.transcript_display.append(
                "<span style='color:#86EFAC;'>Audio device changed — listening refreshed.</span><br>"
            )
            print(
                f"[audio-monitor] Listening restarted - mic={mic_idx}, system={system_idx}."
            )
        else:
            self.window.audio_recorder.stop_recording()
            self.window.record_btn.setText("Listen")
            self.window._set_status("ERROR")
            print("[audio-monitor] No usable input after device refresh.")
