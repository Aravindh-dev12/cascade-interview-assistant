import config
from engine.audio_recorder import AudioRecorder


def install_audio_device_recovery():
    """Validate persisted PortAudio indices before every listening start.

    Windows/PortAudio device indices are not stable across reboots, Bluetooth
    reconnects, USB changes, driver updates, or docking. Persisted indices are
    therefore treated as hints and replaced with current devices when stale.
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

    def toggle_recording(window):
        if not window.audio_recorder.is_recording:
            window._ensure_audio_defaults()
            window.audio_recorder.set_devices(
                window.settings.get("mic_device_idx", -1),
                window.settings.get("system_device_idx", -1),
            )
        return original_toggle_recording(window)

    OverlayWindow._ensure_audio_defaults = ensure_audio_defaults
    OverlayWindow.toggle_recording = toggle_recording
    OverlayWindow._audio_recovery_patch_installed = True
