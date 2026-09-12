import os
import queue
import threading
import time

import config
from engine.nvidia_omni_vision import NvidiaOmniVisionClient


def _float_env(name, default, minimum, maximum):
    try:
        value = float(os.environ.get(name, str(default)))
    except Exception:
        value = float(default)
    return max(float(minimum), min(float(maximum), value))


def install_local_qwen_pipeline():
    """Use local Qwen for every final answer and NVIDIA only for perception.

    Audio path: Parakeet/Riva -> transcript -> local Qwen -> overlay.
    Visual path: manual screen/camera -> NVIDIA Nemotron Omni image-to-text -> local Qwen.
    If NVIDIA vision is busy/unavailable, the same manual image falls back directly to
    Qwen vision instead of ending the request with a cloud-service error.
    """
    from engine import copilot_ai as ai_module
    from PySide6.QtWidgets import QLabel
    from ui.settings_dialog import SettingsDialog

    CopilotAI = ai_module.CopilotAI
    if getattr(CopilotAI, "_local_qwen_pipeline_installed", False):
        return

    # Automatic screenshot inference is gone, so warming the local model no longer
    # races a startup vision request. Keep Qwen ready for Listen/Send/Capture actions.
    os.environ["OLLAMA_WARMUP"] = "1"

    original_init = CopilotAI.__init__
    original_settings_init = SettingsDialog.__init__

    def ai_init(ai, provider=None):
        original_init(ai, provider="ollama")
        ai.provider_preference = "ollama"
        ai.omni_vision = NvidiaOmniVisionClient(
            api_key=os.environ.get("NVIDIA_API_KEY", "")
        )
        print(
            f"[pipeline] Final answer engine: local {ai.ollama.model}; "
            f"vision extractor: {ai.omni_vision.model}"
        )

    def set_config(ai, provider=None):
        settings = config.load_settings()
        local_model = os.environ.get("OLLAMA_MODEL", "").strip() or settings.get(
            "local_model", config.DEFAULT_LOCAL_MODEL
        )
        base_url = os.environ.get("OLLAMA_BASE_URL", "").strip() or settings.get(
            "ollama_base_url", config.DEFAULT_OLLAMA_BASE_URL
        )
        num_ctx = int(
            os.environ.get("OLLAMA_NUM_CTX", settings.get("ollama_num_ctx", 8192))
        )
        ai.ollama.reconfigure(
            base_url=base_url,
            model=local_model,
            num_ctx=num_ctx,
        )
        ai.provider_preference = "ollama"
        if not hasattr(ai, "omni_vision"):
            ai.omni_vision = NvidiaOmniVisionClient()
        else:
            ai.omni_vision.reconfigure()

    def runtime_label(ai):
        ready, _, age = ai.ollama.cached_available()
        if age > 5.0:
            ai.ollama.refresh_async()
        state = ai.ollama.model if ready else f"loading/checking {ai.ollama.model}"
        return f"LOCAL · {state} · NVIDIA vision + local vision fallback"

    def bounded_qwen_stream(ai, prompt, max_tokens, image_bytes_list=None):
        events = queue.Queue()

        def run():
            try:
                produced = False
                for piece in ai._qwen_stream(
                    prompt,
                    max_tokens=max_tokens,
                    image_bytes_list=image_bytes_list,
                ):
                    if not piece:
                        continue
                    produced = True
                    events.put(("chunk", piece))
                events.put(("done", None if produced else "no answer tokens"))
            except Exception as exc:
                events.put(("error", exc))

        threading.Thread(
            target=run,
            daemon=True,
            name="local-qwen-vision" if image_bytes_list else "local-qwen-answer",
        ).start()

        if image_bytes_list:
            # Clamp old .env values so a previous 20s timeout cannot break manual vision.
            first_timeout = _float_env(
                "LOCAL_QWEN_VISION_FIRST_TOKEN_TIMEOUT_SECONDS", 60.0, 60.0, 180.0
            )
        else:
            first_timeout = _float_env(
                "LOCAL_QWEN_FIRST_TOKEN_TIMEOUT_SECONDS", 45.0, 45.0, 120.0
            )
        idle_timeout = _float_env(
            "LOCAL_QWEN_STREAM_IDLE_TIMEOUT_SECONDS", 45.0, 15.0, 120.0
        )
        started = time.monotonic()
        first = True
        while True:
            timeout = first_timeout if first else idle_timeout
            try:
                event_type, payload = events.get(timeout=timeout)
            except queue.Empty:
                phase = "vision first token" if image_bytes_list and first else (
                    "first token" if first else "stream"
                )
                raise RuntimeError(
                    f"Local Qwen {phase} timeout after {timeout:.1f}s. "
                    "Run `ollama ps` to verify qwen3.5:4b is loaded."
                )
            if event_type == "chunk":
                if first:
                    mode = "vision" if image_bytes_list else "text"
                    print(
                        f"[qwen] First {mode} answer token in {time.monotonic() - started:.2f}s · "
                        f"model={ai.ollama.model}"
                    )
                    first = False
                yield payload
            elif event_type == "done":
                if payload and first:
                    raise RuntimeError(f"Local Qwen produced no answer: {payload}")
                return
            elif event_type == "error":
                if isinstance(payload, BaseException):
                    raise RuntimeError(f"Local Qwen failed: {payload}") from payload
                raise RuntimeError(f"Local Qwen failed: {payload}")

    def generate_text_stream(ai, custom_query=None):
        ai.set_config(provider="ollama")
        task = custom_query or (
            "Answer the latest substantive interviewer question in the practice transcript."
        )
        prompt = ai._build_prompt(task)
        max_tokens = int(os.environ.get("TEXT_MAX_TOKENS", "700"))
        print("[pipeline] Transcript/chat -> local Qwen")
        yield from bounded_qwen_stream(ai, prompt, max_tokens=max_tokens)

    def generate_vision_stream(
        ai,
        image_bytes,
        custom_query=None,
        use_image_history=False,
    ):
        ai.set_config(provider="ollama")
        task = custom_query or (
            "Solve or explain the current visible practice question. Give the useful answer first."
        )
        max_tokens = int(os.environ.get("VISION_MAX_TOKENS", "1200"))

        print(
            f"[pipeline] Manual image -> NVIDIA Omni extraction · bytes={len(image_bytes or b'')}"
        )
        try:
            visual_text = ai.omni_vision.extract(image_bytes, task_hint=task)
        except Exception as exc:
            # Hosted vision can transiently return 503/429. A manual Capture action
            # should still produce an answer, so fall back to Qwen's image capability.
            print(f"[vision] NVIDIA extraction unavailable: {exc}")
            print("[vision] Falling back to local Qwen image analysis for this capture.")
            fallback_prompt = ai._build_prompt(
                task,
                image_task=(
                    "Read the supplied screenshot carefully. Detect whether it contains an MCQ, coding "
                    "problem, debugging task, terminal/output question, SQL, diagram, system-design prompt, "
                    "or general question. For MCQ give the option first; for coding give approach, correct "
                    "code, and complexity; otherwise answer the visible question directly. Do not invent "
                    "text that is not readable."
                ),
            )
            yield from bounded_qwen_stream(
                ai,
                fallback_prompt,
                max_tokens=max_tokens,
                image_bytes_list=[image_bytes],
            )
            return

        qwen_task = (
            f"{task}\n\n"
            "NVIDIA NEMOTRON OMNI VISUAL EXTRACTION:\n"
            f"{visual_text}\n\n"
            "Use the visual extraction as evidence from the current screen/camera. "
            "Now produce the final candidate-ready answer. For an MCQ, put the correct option first. "
            "For coding, give the approach, correct code, and time/space complexity. "
            "For debugging, identify the bug and corrected code. Do not claim to see anything beyond "
            "the extracted visual context."
        )
        prompt = ai._build_prompt(qwen_task)
        print("[pipeline] NVIDIA visual text -> local Qwen")
        yield from bounded_qwen_stream(ai, prompt, max_tokens=max_tokens)

    def route_stream(ai, prompt, max_tokens, image_bytes_list=None):
        # Defensive compatibility for legacy callers: final generation is always local.
        if image_bytes_list:
            try:
                visual_text = ai.omni_vision.extract(
                    image_bytes_list[-1],
                    task_hint="Use the current visual context to answer the request.",
                )
                prompt = (
                    prompt
                    + "\n\nNVIDIA NEMOTRON OMNI VISUAL EXTRACTION:\n"
                    + visual_text
                )
                image_bytes_list = None
            except Exception as exc:
                print(f"[vision] NVIDIA compatibility extraction failed: {exc}; using local vision.")
        yield from bounded_qwen_stream(
            ai,
            prompt,
            max_tokens=max_tokens,
            image_bytes_list=image_bytes_list,
        )

    def settings_init(dialog, current_settings, parent=None):
        original_settings_init(dialog, current_settings, parent)
        if hasattr(dialog, "provider_combo"):
            dialog.provider_combo.clear()
            dialog.provider_combo.addItem(
                "Local Qwen 3.5 · Parakeet audio · NVIDIA/local vision",
                "ollama",
            )
            dialog.provider_combo.setCurrentIndex(0)
            dialog.provider_combo.setEnabled(False)

        # The runtime is intentionally manual-first. Disable legacy auto controls so
        # the UI matches actual behavior instead of suggesting background actions.
        if hasattr(dialog, "auto_start_check"):
            dialog.auto_start_check.setChecked(False)
            dialog.auto_start_check.setEnabled(False)
            dialog.auto_start_check.setText("Listening starts only when you click Listen")
        if hasattr(dialog, "screen_watch_check"):
            dialog.screen_watch_check.setChecked(False)
            dialog.screen_watch_check.setEnabled(False)
            dialog.screen_watch_check.setText("Screen capture runs only when you click Capture screen")
        if hasattr(dialog, "screen_answer_check"):
            dialog.screen_answer_check.setChecked(False)
            dialog.screen_answer_check.setEnabled(False)
            dialog.screen_answer_check.setText("Automatic background screen answering is disabled")
        if hasattr(dialog, "include_screen_check"):
            dialog.include_screen_check.setChecked(False)
            dialog.include_screen_check.setEnabled(False)
            dialog.include_screen_check.setText("Voice does not attach screenshots automatically")

        replacements = {
            "NVIDIA Kimi-K3 and local Qwen are the only answer engines. NVIDIA_API_KEY is loaded automatically from the project .env file.":
                "Local Qwen is the final answer engine. NVIDIA_API_KEY is loaded from .env for Parakeet speech-to-text and Nemotron Omni image-to-text.",
            "Hybrid gives Kimi-K3 a short head start and races local Qwen when cloud latency is high. No API key is stored in Settings.":
                "Local Qwen 3.5 generates every final answer. NVIDIA is used only for speech transcription and visual extraction. No API key is stored in Settings.",
            "Local fallback: install Ollama and run  ollama pull qwen3.5:4b. The app keeps Qwen warm and streams the first available answer.":
                "Answer engine: install Ollama and run  ollama pull qwen3.5:4b. The app warms Qwen at startup for faster manual responses.",
            "NVIDIA Riva/Nemotron uses the same NVIDIA_API_KEY from .env for streaming transcription.":
                "NVIDIA Parakeet CTC uses the NVIDIA_API_KEY from .env for streaming transcription.",
        }
        for label in dialog.findChildren(QLabel):
            text = label.text()
            if text in replacements:
                label.setText(replacements[text])

    CopilotAI.__init__ = ai_init
    CopilotAI.set_config = set_config
    CopilotAI.runtime_label = runtime_label
    CopilotAI._route_stream = route_stream
    CopilotAI.generate_text_stream = generate_text_stream
    CopilotAI.generate_vision_stream = generate_vision_stream
    SettingsDialog.__init__ = settings_init
    CopilotAI._local_qwen_pipeline_installed = True
