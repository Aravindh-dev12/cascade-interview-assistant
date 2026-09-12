import os
import time


def _env_enabled(name, default=True):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def install_visual_autopilot():
    """Keep visual context fresh without starving real-time spoken answers.

    While live listening is active, stable screen frames are cached continuously but
    are not submitted on their own. A spoken question attaches the fresh visual only
    when the question actually refers to the visible problem/code/MCQ/camera. Manual
    Capture and Camera actions always force an immediate vision request.
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

    def _screen_request_pending(window):
        if any(item.get("kind") == "screen" for item in window.requests.values()):
            return True
        return any(item.get("kind") == "screen" for item in window.request_queue)

    def _supersede_auto_screen_requests(window):
        window.request_queue = [
            item for item in window.request_queue if item.get("kind") != "screen"
        ]
        for request_id, active in list(window.requests.items()):
            if active.get("kind") == "screen":
                window.stale_request_ids.add(request_id)

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
                    f"[vision] Interviewer prompt needs visual context · "
                    f"age={now - window.latest_screen_time:.2f}s"
                )
            elif visual_ready:
                print("[vision] Interviewer prompt routed text-only; cached screen not attached.")
        return original_handle_transcription(window, speaker, text)

    def handle_screen_frame(window, image_bytes):
        window.latest_screen_bytes = image_bytes
        window.latest_screen_time = time.monotonic()
        window.screen_meta.setText("SCREEN CONTEXT LIVE")

        # During an interview, screen capture remains live but inference is driven by
        # interviewer intent. This prevents screenshots from flooding Kimi/Qwen and
        # starving a simple spoken question such as "Tell me about yourself".
        if window.audio_recorder.is_recording and not _env_enabled(
            "AUTO_VISUAL_WHILE_LISTENING", False
        ):
            return

        if not _env_enabled("AUTO_VISUAL_ANSWER", True):
            return
        if not window.settings.get("auto_answer_screen", True):
            return
        if not overlay_module._practice_mode_enabled():
            return
        if _screen_request_pending(window):
            return

        now = time.monotonic()
        cooldown = max(
            1.5,
            float(os.environ.get("VISUAL_AUTO_ANSWER_COOLDOWN_SECONDS", "4.0")),
        )
        if now - window.last_screen_answer_time < cooldown:
            return
        window.last_screen_answer_time = now

        print(
            f"[vision] Stable screen frame -> AI · screen-only · "
            f"bytes={len(image_bytes)} · frames=1"
        )
        window._enqueue_ai(
            source="Visible question",
            kind="screen",
            image_bytes=image_bytes,
            custom_query=(
                "Inspect the visible practice question and answer it now. "
                "If it is an MCQ, put the correct option/answer first and then one short reason. "
                "If it is a coding problem, state the approach briefly, then provide correct code and complexity. "
                "If it is debugging, identify the bug and give the corrected code. "
                "If it is a conceptual, system-design, diagram, terminal, SQL, or output question, answer directly. "
                "If no actual question is visible, do not invent one; reply only NO_QUESTION_VISIBLE."
            ),
            use_image_history=False,
        )

    def submit_screen_capture(window, image_bytes, source="Manual screen capture"):
        window.latest_screen_bytes = image_bytes
        window.latest_screen_time = time.monotonic()
        _supersede_auto_screen_requests(window)
        print(f"[vision] Manual screen capture -> AI · bytes={len(image_bytes)}")
        window._enqueue_ai(
            source=source,
            kind="manual_screen",
            image_bytes=image_bytes,
            custom_query=(
                "Analyze the captured practice question completely. "
                "For MCQs, give the correct option first. For coding, give the approach, correct code, "
                "and time/space complexity. For debugging, errors, SQL, output, diagrams, or conceptual "
                "questions, give the direct answer first. Read all visible text carefully."
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
            _supersede_auto_screen_requests(controller.window)
            print(
                f"[vision] Camera frame -> AI · bytes={len(controller.latest_camera_bytes)}"
            )
        return original_analyze_camera(controller)

    def settings_init(dialog, current_settings, parent=None):
        original_settings_init(dialog, current_settings, parent)
        if hasattr(dialog, "screen_answer_check"):
            dialog.screen_answer_check.setText(
                "Automatically analyze stable visible questions when not listening"
            )
        if hasattr(dialog, "include_screen_check"):
            dialog.include_screen_check.setText(
                "Attach fresh screen/camera only when the interviewer refers to it"
            )

    OverlayWindow.handle_transcription = handle_transcription
    OverlayWindow.handle_screen_frame = handle_screen_frame
    OverlayWindow.submit_screen_capture = submit_screen_capture
    CameraVisionControls.__init__ = camera_init
    CameraVisionControls.analyze_camera = analyze_camera
    SettingsDialog.__init__ = settings_init
    OverlayWindow._visual_autopilot_installed = True
