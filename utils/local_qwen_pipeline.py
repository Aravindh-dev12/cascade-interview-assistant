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
    Visual path: screen/camera -> NVIDIA Nemotron Omni image-to-text -> local Qwen -> overlay.
    Kimi is intentionally not used by this runtime pipeline.
    """
    from engine import copilot_ai as ai_module
    from PySide6.QtWidgets import QLabel
    from ui.settings_dialog import SettingsDialog

    CopilotAI = ai_module.CopilotAI
    if getattr(CopilotAI, "_local_qwen_pipeline_installed", False):
        return

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
        return f"LOCAL · {state} · NVIDIA vision"

    def bounded_qwen_stream(ai, prompt, max_tokens):
        events = queue.Queue()

        def run():
            try:
                produced = False
                for piece in ai._qwen_stream(
                    prompt,
                    max_tokens=max_tokens,
                    image_bytes_list=None,
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
            name="local-qwen-answer",
        ).start()

        first_timeout = _float_env(
            "LOCAL_QWEN_FIRST_TOKEN_TIMEOUT_SECONDS", 20.0, 3.0, 120.0
        )
        idle_timeout = _float_env(
            "LOCAL_QWEN_STREAM_IDLE_TIMEOUT_SECONDS", 30.0, 5.0, 120.0
        )
        started = time.monotonic()
        first = True
        while True:
            timeout = first_timeout if first else idle_timeout
            try:
                event_type, payload = events.get(timeout=timeout)
            except queue.Empty:
                phase = "first token" if first else "stream"
                raise RuntimeError(
                    f"Local Qwen {phase} timeout after {timeout:.1f}s. "
                    "Check `ollama ps` and GPU/CPU load."
                )
            if event_type == "chunk":
                if first:
                    print(
                        f"[qwen] First answer token in {time.monotonic() - started:.2f}s · "
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
        print("[pipeline] Parakeet/transcript text -> local Qwen")
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
        print(
            f"[pipeline] Image -> NVIDIA Omni text extraction · bytes={len(image_bytes or b'')}"
        )
        visual_text = ai.omni_vision.extract(image_bytes, task_hint=task)
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
        max_tokens = int(os.environ.get("VISION_MAX_TOKENS", "1200"))
        print("[pipeline] NVIDIA visual text -> local Qwen")
        yield from bounded_qwen_stream(ai, prompt, max_tokens=max_tokens)

    def route_stream(ai, prompt, max_tokens, image_bytes_list=None):
        # Defensive compatibility for legacy callers: final generation is always local.
        if image_bytes_list:
            visual_text = ai.omni_vision.extract(
                image_bytes_list[-1],
                task_hint="Use the current visual context to answer the request.",
            )
            prompt = (
                prompt
                + "\n\nNVIDIA NEMOTRON OMNI VISUAL EXTRACTION:\n"
                + visual_text
            )
        yield from bounded_qwen_stream(ai, prompt, max_tokens=max_tokens)

    def settings_init(dialog, current_settings, parent=None):
        original_settings_init(dialog, current_settings, parent)
        if hasattr(dialog, "provider_combo"):
            dialog.provider_combo.clear()
            dialog.provider_combo.addItem(
                "Local Qwen 3.5 · NVIDIA Parakeet audio + Nemotron Omni vision",
                "ollama",
            )
            dialog.provider_combo.setCurrentIndex(0)
            dialog.provider_combo.setEnabled(False)

        replacements = {
            "NVIDIA Kimi-K3 and local Qwen are the only answer engines. NVIDIA_API_KEY is loaded automatically from the project .env file.":
                "Local Qwen is the final answer engine. NVIDIA_API_KEY is loaded from .env for Parakeet speech-to-text and Nemotron Omni image-to-text.",
            "Hybrid gives Kimi-K3 a short head start and races local Qwen when cloud latency is high. No API key is stored in Settings.":
                "Local Qwen 3.5 generates every final answer. NVIDIA is used only for speech transcription and visual extraction. No API key is stored in Settings.",
            "Local fallback: install Ollama and run  ollama pull qwen3.5:4b. The app keeps Qwen warm and streams the first available answer.":
                "Answer engine: install Ollama and run  ollama pull qwen3.5:4b. Parakeet transcripts and NVIDIA visual text are sent to this local model.",
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
