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
    If NVIDIA vision is busy, unavailable, or returns an unusably short extraction, the
    same manual image falls back directly to Qwen vision instead of ending the request.
    """
    from engine import copilot_ai as ai_module
    from PySide6.QtWidgets import QLabel
    from ui.settings_dialog import SettingsDialog

    CopilotAI = ai_module.CopilotAI
    if getattr(CopilotAI, "_local_qwen_pipeline_installed", False):
        return

    # Manual-first mode has no startup screenshot inference, so pre-warm Qwen.
    os.environ["OLLAMA_WARMUP"] = "1"

    original_init = CopilotAI.__init__
    original_settings_init = SettingsDialog.__init__

    def ai_init(ai, provider=None):
        original_init(ai, provider="ollama")
        ai.provider_preference = "ollama"
        ai.omni_vision = NvidiaOmniVisionClient(
            api_key=os.environ.get("NVIDIA_API_KEY", "")
        )
        ai._qwen_prepare_lock = threading.Lock()
        ai._qwen_ready_for_answers = False
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
        previous = (ai.ollama.base_url, ai.ollama.model, ai.ollama.num_ctx)
        ai.ollama.reconfigure(
            base_url=base_url,
            model=local_model,
            num_ctx=num_ctx,
        )
        current = (ai.ollama.base_url, ai.ollama.model, ai.ollama.num_ctx)
        if previous != current:
            ai._qwen_ready_for_answers = False
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

    def ensure_qwen_ready(ai):
        """Verify the configured model exists and synchronously prepare it before use."""
        ready, detail = ai.ollama.available(cache_seconds=0)
        if not ready:
            raise RuntimeError(
                f"Local Qwen model `{ai.ollama.model}` is not installed/available: {detail}. "
                f"Run `ollama pull {ai.ollama.model}` and then retry."
            )

        lock = getattr(ai, "_qwen_prepare_lock", None)
        if lock is None:
            lock = threading.Lock()
            ai._qwen_prepare_lock = lock

        with lock:
            if getattr(ai, "_qwen_ready_for_answers", False):
                return

            # The normal startup warmup may already have completed; a second tiny one-token
            # warmup is harmless and guarantees the first real request is not the cold load.
            print(f"[ollama] Preparing local model before answer: {ai.ollama.model}")
            if not ai.ollama.warmup():
                ready, detail = ai.ollama.available(cache_seconds=0)
                if not ready:
                    raise RuntimeError(
                        f"Local Qwen model `{ai.ollama.model}` could not be prepared: {detail}. "
                        f"Run `ollama pull {ai.ollama.model}` and verify Ollama is running."
                    )
                raise RuntimeError(
                    f"Local Qwen model `{ai.ollama.model}` is installed but warmup failed. "
                    "Run `ollama run qwen3.5:4b \"Reply only READY\"` once and inspect `ollama ps`."
                )
            ai._qwen_ready_for_answers = True

    def validate_visual_text(text):
        cleaned = str(text or "").strip()
        try:
            configured_min = int(os.environ.get("NVIDIA_VISION_MIN_USEFUL_CHARS", "32"))
        except Exception:
            configured_min = 32
        minimum = max(12, min(200, configured_min))
        label_only = {
            "mcq",
            "coding",
            "debug",
            "debugging",
            "general",
            "image",
            "screen",
            "unknown",
            "none",
        }
        if len(cleaned) < minimum or cleaned.lower().strip(" .:-") in label_only:
            raise RuntimeError(
                f"NVIDIA Omni extraction was too short to answer safely "
                f"({len(cleaned)} chars; need at least {minimum})."
            )
        return cleaned

    def bounded_qwen_stream(ai, prompt, max_tokens, image_bytes_list=None):
        # Do this before the answer thread starts. If Ollama/model setup is the problem,
        # the UI gets a concrete error instead of appearing to do nothing for a minute.
        ensure_qwen_ready(ai)
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
            first_timeout = _float_env(
                "LOCAL_QWEN_VISION_FIRST_TOKEN_TIMEOUT_SECONDS", 120.0, 90.0, 240.0
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
                    "Run `ollama ps` while this request is active and inspect PROCESSOR."
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
            visual_text = validate_visual_text(visual_text)
        except Exception as exc:
            print(f"[vision] NVIDIA extraction unusable/unavailable: {exc}")
            print("[vision] Falling back to local Qwen image analysis using the original screenshot.")
            fallback_prompt = ai._build_prompt(
                task,
                image_task=(
                    "Read the supplied screenshot carefully. Detect whether it contains an MCQ, coding "
                    "problem, debugging task, terminal/output question, SQL, diagram, system-design prompt, "
                    "or general question. For an MCQ give the correct option first; for coding give the "
                    "approach, correct code, and complexity; otherwise answer the visible question directly. "
                    "Do not invent text that is not readable."
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
                visual_text = validate_visual_text(visual_text)
                prompt = (
                    prompt
                    + "\n\nNVIDIA NEMOTRON OMNI VISUAL EXTRACTION:\n"
                    + visual_text
                )
                image_bytes_list = None
            except Exception as exc:
                print(
                    f"[vision] NVIDIA compatibility extraction unusable/unavailable: {exc}; "
                    "using local vision."
                )
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
