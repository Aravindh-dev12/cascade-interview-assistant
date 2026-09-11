import os
import queue
import threading
import time

import config
from engine.audio_recorder import AudioRecorder
from engine.copilot_ai import SYSTEM_PROMPT


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


def install_fast_hybrid_failover(window):
    """Hedge Kimi-K3 with local Qwen instead of waiting serially for failure.

    Kimi gets a short head start. If it has not emitted a token by the hedge
    deadline, Qwen starts concurrently. The first provider to emit answer text
    wins the request. This keeps cloud quality when Kimi is responsive while
    preventing a transient cloud stall from blocking the desktop UI.
    """
    if getattr(window, "_fast_hybrid_failover_installed", False):
        return

    previous_stream = window.copilot_ai._local_or_fallback_stream

    def requested_provider():
        provider = (
            os.environ.get("AI_PROVIDER", "").strip().lower()
            or str(window.settings.get("ai_provider", "hybrid") or "hybrid").strip().lower()
        )
        if provider == "auto" and os.environ.get("NVIDIA_API_KEY", "").strip():
            return "hybrid"
        return provider

    def run_provider(name, factory, events):
        try:
            produced = False
            for piece in factory():
                if not piece:
                    continue
                produced = True
                events.put((name, "chunk", piece))
            events.put((name, "done", None if produced else "no answer tokens"))
        except Exception as exc:
            events.put((name, "error", exc))

    def qwen_factory(prompt, max_tokens, image_bytes_list):
        def generate():
            ready, detail = window.copilot_ai.ollama.available(cache_seconds=0)
            if not ready:
                raise RuntimeError(f"local Qwen unavailable: {detail}")
            yield from window.copilot_ai.ollama.chat_stream(
                prompt,
                max_tokens=max_tokens,
                image_bytes_list=image_bytes_list,
            )

        return generate

    def kimi_factory(prompt, max_tokens, image_bytes_list):
        def generate():
            kimi = getattr(window, "nvidia_kimi", None)
            if kimi is None:
                raise RuntimeError("NVIDIA Kimi client is not initialized")
            kimi.reconfigure(api_key=os.environ.get("NVIDIA_API_KEY", "").strip())
            if not kimi.available():
                raise RuntimeError("NVIDIA Kimi credentials/configuration are unavailable")
            yield from kimi.chat_stream(
                prompt,
                max_tokens=max_tokens,
                image_bytes_list=image_bytes_list,
                system_prompt=SYSTEM_PROMPT,
            )

        return generate

    def hedged_stream(prompt, max_tokens, image_bytes_list=None):
        if requested_provider() != "hybrid":
            yield from previous_stream(
                prompt,
                max_tokens=max_tokens,
                image_bytes_list=image_bytes_list,
            )
            return

        started = time.monotonic()
        events = queue.Queue()
        errors = {}
        done = set()
        winner = None
        qwen_started = False
        hedge_seconds = max(
            0.35,
            min(4.0, float(os.environ.get("HYBRID_HEDGE_SECONDS", "1.25"))),
        )

        print(
            f"[hybrid] Request started · visual={bool(image_bytes_list)} · "
            f"Kimi head-start={hedge_seconds:.2f}s"
        )
        threading.Thread(
            target=run_provider,
            args=("kimi", kimi_factory(prompt, max_tokens, image_bytes_list), events),
            daemon=True,
            name="hybrid-kimi",
        ).start()

        try:
            first = events.get(timeout=hedge_seconds)
            events.put(first)
        except queue.Empty:
            pass

        while winner is None:
            if not qwen_started:
                should_start_qwen = time.monotonic() - started >= hedge_seconds or "kimi" in errors or "kimi" in done
                if should_start_qwen:
                    qwen_started = True
                    print("[hybrid] Kimi has no first token yet; starting local Qwen hedge.")
                    threading.Thread(
                        target=run_provider,
                        args=("qwen", qwen_factory(prompt, max_tokens, image_bytes_list), events),
                        daemon=True,
                        name="hybrid-qwen",
                    ).start()

            try:
                name, event_type, payload = events.get(timeout=0.10)
            except queue.Empty:
                if not qwen_started and time.monotonic() - started >= hedge_seconds:
                    continue
                if qwen_started and len(errors) + len(done) >= 2:
                    break
                continue

            if event_type == "chunk":
                winner = name
                window.copilot_ai._active_provider = "nvidia" if name == "kimi" else "ollama"
                print(f"[hybrid] {name.upper()} won first token in {time.monotonic() - started:.2f}s")
                yield payload
                break
            if event_type == "error":
                errors[name] = payload
                print(f"[hybrid] {name} failed before first token: {payload}")
            elif event_type == "done":
                done.add(name)
                if payload:
                    errors[name] = RuntimeError(str(payload))

        if winner is None:
            details = "; ".join(f"{name}: {err}" for name, err in errors.items()) or "no provider produced output"
            raise RuntimeError(f"Hybrid inference failed: {details}")

        while True:
            name, event_type, payload = events.get()
            if name != winner:
                continue
            if event_type == "chunk":
                yield payload
            elif event_type == "done":
                return
            elif event_type == "error":
                print(f"[hybrid] {winner} stream ended after partial output: {payload}")
                return

    window.copilot_ai._local_or_fallback_stream = hedged_stream
    window._fast_hybrid_failover_installed = True
