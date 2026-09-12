import os
import time


def _env_enabled(name, default=True):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def install_visual_autopilot():
    """Route captured visual context into practice answers.

    The current runtime is manual-first. Manual Capture is scroll-aware: repeated
    captures contribute to one current visual problem so long questions can be
    reconstructed across several scroll positions. The Clear button resets that
    accumulated visual context together with the rest of the session.
    """
    from ui import overlay_window as overlay_module
    from ui.settings_dialog import SettingsDialog
    from utils.realtime_multimodal import CameraVisionControls

    OverlayWindow = overlay_module.OverlayWindow
    if getattr(OverlayWindow, "_visual_autopilot_installed", False):
        return

    original_handle_transcription = OverlayWindow.handle_transcription
    original_camera_init = CameraVisionControls.__init__
    original_analyze_camera = CameraVisionControls.analyze_camera
    original_settings_init = SettingsDialog.__init__

    def _drop_queued_screen_requests(window, include_manual=False):
        kinds = {"screen"}
        if include_manual:
            kinds.add("manual_screen")
        window.request_queue = [
            item for item in window.request_queue if item.get("kind") not in kinds
        ]

    def handle_transcription(window, speaker, text):
        if speaker == "Interviewer" and overlay_module.is_substantive_question(text):
            now = time.monotonic()
            max_age = float(window.settings.get("screen_context_max_age_seconds", 12.0))
            visual_ready = bool(
                window.settings.get("include_screen_with_speech", False)
                and window.latest_screen_bytes
                and now - window.latest_screen_time <= max_age
            )
            needs_visual = overlay_module.should_attach_screen(text)
            if visual_ready and needs_visual:
                print(
                    f"[vision] Interviewer prompt needs current screen context · "
                    f"age={now - window.latest_screen_time:.2f}s"
                )
            elif visual_ready:
                print("[vision] Spoken question routed text-only; screen is analyzed independently.")
        return original_handle_transcription(window, speaker, text)

    def handle_screen_frame(window, image_bytes):
        window.latest_screen_bytes = image_bytes
        window.latest_screen_time = time.monotonic()
        window.screen_meta.setText("SCREEN ANALYSIS LIVE")

        if not _env_enabled("AUTO_VISUAL_ANSWER", False):
            return
        if not window.settings.get("auto_answer_screen", False):
            return
        if not overlay_module._practice_mode_enabled():
            return

        now = time.monotonic()
        cooldown = max(
            0.35,
            float(os.environ.get("VISUAL_AUTO_ANSWER_COOLDOWN_SECONDS", "0.75")),
        )
        if now - window.last_screen_answer_time < cooldown:
            return
        window.last_screen_answer_time = now

        _drop_queued_screen_requests(window)
        print(
            f"[vision] Captured stable screen -> analyze now · bytes={len(image_bytes)}"
        )
        window._enqueue_ai(
            source="Screen question",
            kind="screen",
            image_bytes=image_bytes,
            custom_query=(
                "Analyze the visible practice question. For an MCQ, give the correct option first and explain briefly. "
                "For coding, use the currently visible selected/requested programming language; default to Python 3 "
                "only if none is visible. Give a short approach, complete code, time/space complexity, and key edge "
                "cases. For debugging, return corrected code. Do not invent unreadable text."
            ),
            use_image_history=False,
        )

    def submit_screen_capture(window, image_bytes, source="Manual screen capture"):
        window.latest_screen_bytes = image_bytes
        window.latest_screen_time = time.monotonic()

        # Manual scrolling can create multiple captures. Keep at most the newest waiting
        # request while one is active, but make every executed manual request contribute
        # to CopilotAI's visual history.
        active_manual = any(
            request.get("kind") == "manual_screen"
            for request in window.requests.values()
        )
        _drop_queued_screen_requests(window, include_manual=True)
        if active_manual:
            print(
                "[vision] A screenshot is already being analyzed; keeping this as the newest queued scroll capture."
            )

        print(
            f"[vision] Manual screen capture -> analyze now · bytes={len(image_bytes)} · scroll-aware=true"
        )
        window._enqueue_ai(
            source=source,
            kind="manual_screen",
            image_bytes=image_bytes,
            custom_query=(
                "Solve the current visible practice question using this capture plus any earlier captures from the "
                "same scrolling session. If this is an MCQ, put the correct option and answer first, then a concise "
                "reason. If this is coding, honor the currently selected/requested language shown on screen; use "
                "Python 3 only when no language is visible. Give a short approach, complete platform-compatible code, "
                "time and space complexity, and important correctness/edge cases so the solution is designed for "
                "hidden tests rather than only samples. If the question is incomplete because more content is below, "
                "use previous scroll captures and say NEED_MORE_SCREEN only when a required part is genuinely missing."
            ),
            use_image_history=True,
        )

    def camera_init(controller, window):
        original_camera_init(controller, window)
        if window.latest_screen_bytes and not controller.latest_screen_bytes:
            controller.latest_screen_bytes = window.latest_screen_bytes
            controller.latest_screen_time = window.latest_screen_time
            controller._publish_combined_context()

    def analyze_camera(controller):
        if controller.latest_camera_bytes:
            _drop_queued_screen_requests(controller.window, include_manual=True)
            print(
                f"[vision] Camera frame -> analyze now · bytes={len(controller.latest_camera_bytes)}"
            )
        return original_analyze_camera(controller)

    def settings_init(dialog, current_settings, parent=None):
        original_settings_init(dialog, current_settings, parent)
        if hasattr(dialog, "screen_answer_check"):
            dialog.screen_answer_check.setText(
                "Background screen answering is disabled; use Capture screen"
            )
        if hasattr(dialog, "include_screen_check"):
            dialog.include_screen_check.setText(
                "Attach a recent screen only when a spoken question explicitly refers to it"
            )

    OverlayWindow.handle_transcription = handle_transcription
    OverlayWindow.handle_screen_frame = handle_screen_frame
    OverlayWindow.submit_screen_capture = submit_screen_capture
    CameraVisionControls.__init__ = camera_init
    CameraVisionControls.analyze_camera = analyze_camera
    SettingsDialog.__init__ = settings_init
    OverlayWindow._visual_autopilot_installed = True
