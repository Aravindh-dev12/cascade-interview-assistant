import config
from engine.audio_recorder import AudioRecorder


def install_audio_device_recovery():
    """Validate persisted audio routing and recover devices that fail at open time.

    Windows/PortAudio device indices are not stable, and some stale WDM/KS entries
    can still enumerate as valid even though opening the stream fails. We therefore
    validate indices first, then perform one automatic retry with the current
    Windows default microphone when the selected microphone cannot actually start.
    """
    from ui.overlay_window import OverlayWindow

    if getattr(OverlayWindow, "_audio_recovery_patch_installed", False):
        return

    original_toggle_recording = OverlayWindow.toggle_recording

    def ensure_audio_defaults(window):
        current_mic = int(window.settings.get("mic_device_idx", -1))
        current_system = int(window.settings.get("system_device_idx", -1))

        try:
            mics, loopbacks = AudioRecorder.list_devices()
            valid_mics = {int(item.get("index", -1)) for item in mics}
            valid_system = {int(item.get("index", -1)) for item in loopbacks}
            auto_mic, auto_system = AudioRecorder.auto_detect_devices()
        except Exception as exc:
            print(f"[audio] Device validation failed: {exc}")
            return

        next_mic = current_mic
        next_system = current_system

        if current_mic >= 0 and current_mic not in valid_mics:
            next_mic = int(auto_mic)
        elif current_mic < 0 and int(auto_mic) >= 0:
            next_mic = int(auto_mic)

        if current_system not in valid_system:
            next_system = int(auto_system)

        changed = (next_mic, next_system) != (current_mic, current_system)
        if changed:
            print(
                "[audio] Recovered stale audio routing: "
                f"mic {current_mic} -> {next_mic}, system {current_system} -> {next_system}"
            )
            window.settings["mic_device_idx"] = next_mic
            window.settings["system_device_idx"] = next_system
            window.audio_recorder.set_devices(next_mic, next_system)
            config.save_settings(window.settings)

    def _retry_failed_microphone(window):
        recorder = window.audio_recorder
        selected_mic = int(window.settings.get("mic_device_idx", -1))
        if selected_mic < 0 or recorder.mic_stream is not None:
            return False

        try:
            auto_mic, _ = AudioRecorder.auto_detect_devices()
            auto_mic = int(auto_mic)
        except Exception as exc:
            print(f"[audio] Could not recover failed microphone: {exc}")
            return False

        if auto_mic < 0 or auto_mic == selected_mic:
            print(
                f"[audio] Microphone {selected_mic} failed to open and no different "
                "default microphone is available."
            )
            return False

        system_idx = int(window.settings.get("system_device_idx", -1))
        print(
            f"[audio] Microphone {selected_mic} enumerated but failed to open; "
            f"retrying with current default mic {auto_mic}."
        )

        # The system loopback may already have started successfully. Restart both
        # streams once so STT receives a clean pair of queues and the recovered mic.
        if recorder.is_recording:
            recorder.stop_recording()

        window.settings["mic_device_idx"] = auto_mic
        recorder.set_devices(auto_mic, system_idx)
        config.save_settings(window.settings)
        original_toggle_recording(window)

        if recorder.mic_stream is not None:
            print(f"[audio] Microphone recovery succeeded · mic={auto_mic}")
            return True

        print(f"[audio] Default microphone {auto_mic} also failed to open.")
        return False

    def toggle_recording(window):
        starting = not window.audio_recorder.is_recording
        if starting:
            window._ensure_audio_defaults()
            window.audio_recorder.set_devices(
                window.settings.get("mic_device_idx", -1),
                window.settings.get("system_device_idx", -1),
            )

        result = original_toggle_recording(window)

        if starting and window.audio_recorder.is_recording:
            _retry_failed_microphone(window)
        return result

    OverlayWindow._ensure_audio_defaults = ensure_audio_defaults
    OverlayWindow.toggle_recording = toggle_recording
    OverlayWindow._audio_recovery_patch_installed = True
