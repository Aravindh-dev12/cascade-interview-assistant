import os
import time


def _env_enabled(name, default=True):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def install_visual_autopilot():
    """Route captured visual context into practice answers.

    The current runtime is manual-first, so automatic screen watching is disabled by
    config. This compatibility layer still owns the manual Capture path and ensures
    repeated clicks keep only the newest waiting screenshot instead of building a queue.
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
                window.settings.get("include_screen_with_speech", True)
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

        if not _env_enabled("AUTO_VISUAL_ANSWER", True):
            return
        if not window.settings.get("auto_answer_screen", True):
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
        mode = "while listening" if window.audio_recorder.is_recording else "screen-only"
        print(
            f"[vision] Captured stable screen -> analyze now · {mode} · "
            f"bytes={len(image_bytes)}"
        )
        window._enqueue_ai(
            source="Screen question",
            kind="screen",
            image_bytes=image_bytes,
            custom_query=(
                "Analyze exactly what is visible in this captured practice screen and answer it now. "
                "First determine the problem type from the image. "
                "For an MCQ, return the correct option and answer first, followed by a short reason. "
                "For a coding question, identify the requested language when visible, give the approach, "
                "complete correct code, and time/space complexity. "
                "For debugging, identify the defect and provide corrected code. "
                "For a simple technical, conceptual, aptitude, math, SQL, terminal, output, diagram, "
                "or system-design question, give the direct answer first. "
                "Read visible text/options/code carefully. If no answerable question is actually visible, "
                "reply only NO_QUESTION_VISIBLE."
            ),
            use_image_history=False,
        )

    def submit_screen_capture(window, image_bytes, source="Manual screen capture"):
        window.latest_screen_bytes = image_bytes
        window.latest_screen_time = time.monotonic()

        # A manual capture is user intent. Remove older waiting screen/manual-screen
        # requests so one slow request cannot cause several stale HackerRank captures
        # to execute later. The active worker may finish; only the newest waiting frame remains.
        active_manual = any(
            request.get("kind") == "manual_screen"
            for request in window.requests.values()
        )
        _drop_queued_screen_requests(window, include_manual=True)
        if active_manual:
            print(
                "[vision] A screenshot is already being analyzed; keeping this as the newest queued capture."
            )

        print(f"[vision] Manual screen capture -> analyze now · bytes={len(image_bytes)}")
        window._enqueue_ai(
            source=source,
            kind="manual_screen",
            image_bytes=image_bytes,
            custom_query=(
                "Analyze exactly what is visible in this screenshot and give the final answer now. "
                "Classify it as MCQ, coding, debugging, simple/general question, math, SQL, terminal/output, "
                "diagram, or system design. For MCQs give the correct option first. For coding give approach, "
                "complete code, and time/space complexity. For debugging give the corrected code. "
                "For all other visible questions give the direct answer first. Do not invent unreadable text."
            ),
            use_image_history=False,
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
                "Immediately answer every stable visible practice question"
            )
        if hasattr(dialog, "include_screen_check"):
            dialog.include_screen_check.setText(
                "Also attach the current screen when a spoken question explicitly refers to it"
            )

    OverlayWindow.handle_transcription = handle_transcription
    OverlayWindow.handle_screen_frame = handle_screen_frame
    OverlayWindow.submit_screen_capture = submit_screen_capture
    CameraVisionControls.__init__ = camera_init
    CameraVisionControls.analyze_camera = analyze_camera
    SettingsDialog.__init__ = settings_init
    OverlayWindow._visual_autopilot_installed = True
