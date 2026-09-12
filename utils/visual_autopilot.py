import os
import time


def _env_enabled(name, default=True):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def install_visual_autopilot():
    """Make fresh screen/camera context participate automatically in practice answers.

    The base overlay historically analyzed stable screen frames only while listening
    was stopped and attached visuals to speech only when a small keyword heuristic
    matched. Real ASR often omits words/punctuation, so visible MCQs/coding questions
    could be captured but never sent to Kimi/Qwen. This patch keeps the existing
    queue/scheduling while making visual context proactive and observable.
    """
    from ui import overlay_window as overlay_module

    OverlayWindow = overlay_module.OverlayWindow
    if getattr(OverlayWindow, "_visual_autopilot_installed", False):
        return

    original_should_attach = overlay_module.should_attach_screen
    original_handle_transcription = OverlayWindow.handle_transcription

    def should_attach_visual(text):
        if _env_enabled("ALWAYS_ATTACH_FRESH_VISUAL", True):
            return True
        return original_should_attach(text)

    # The existing overlay already enforces freshness with screen_context_max_age_seconds.
    # Making this predicate true causes fresh SCREEN/CAMERA context to accompany every
    # substantive interviewer question without changing the queue implementation.
    overlay_module.should_attach_screen = should_attach_visual

    def handle_transcription(window, speaker, text):
        if speaker == "Interviewer" and overlay_module.is_substantive_question(text):
            now = time.monotonic()
            max_age = float(window.settings.get("screen_context_max_age_seconds", 12.0))
            visual_ready = bool(
                window.settings.get("include_screen_with_speech", True)
                and window.latest_screen_bytes
                and now - window.latest_screen_time <= max_age
            )
            if visual_ready and _env_enabled("ALWAYS_ATTACH_FRESH_VISUAL", True):
                print(
                    f"[vision] Fresh visual context attached to interviewer question · "
                    f"age={now - window.latest_screen_time:.2f}s"
                )
        return original_handle_transcription(window, speaker, text)

    def handle_screen_frame(window, image_bytes):
        window.latest_screen_bytes = image_bytes
        window.latest_screen_time = time.monotonic()
        window.screen_meta.setText("SCREEN CONTEXT LIVE")

        if not _env_enabled("AUTO_VISUAL_ANSWER", True):
            return
        if not window.settings.get("auto_answer_screen", True):
            return
        if not overlay_module._practice_mode_enabled():
            return

        now = time.monotonic()
        cooldown = max(
            0.8,
            float(os.environ.get("VISUAL_AUTO_ANSWER_COOLDOWN_SECONDS", "2.0")),
        )
        if now - window.last_screen_answer_time < cooldown:
            return
        window.last_screen_answer_time = now

        mode = "while listening" if window.audio_recorder.is_recording else "screen-only"
        print(
            f"[vision] Stable screen frame -> AI · {mode} · "
            f"bytes={len(image_bytes)}"
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
            use_image_history=True,
        )

    def submit_screen_capture(window, image_bytes, source="Manual screen capture"):
        window.latest_screen_bytes = image_bytes
        window.latest_screen_time = time.monotonic()
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
            use_image_history=True,
        )

    OverlayWindow.handle_transcription = handle_transcription
    OverlayWindow.handle_screen_frame = handle_screen_frame
    OverlayWindow.submit_screen_capture = submit_screen_capture
    OverlayWindow._visual_autopilot_installed = True
