import threading

from engine.copilot_ai import CopilotAI, SYSTEM_PROMPT
from engine.nvidia_kimi import NvidiaKimiClient


def install_kimi_runtime():
    """Teach the existing CopilotAI class about NVIDIA Kimi-K3 hybrid routing."""
    if getattr(CopilotAI, "_kimi_runtime_patch_installed", False):
        return

    original_init = CopilotAI.__init__
    original_local_or_fallback = CopilotAI._local_or_fallback_stream

    def normalize_provider(provider):
        provider = str(provider or "auto").strip().lower()
        return provider if provider in {"auto", "hybrid", "nvidia", "ollama", "gemini"} else "hybrid"

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.kimi = NvidiaKimiClient()
        if self.provider_preference == "hybrid":
            threading.Thread(
                target=self.ollama.warmup,
                daemon=True,
                name="ollama-hybrid-warmup",
            ).start()

    def provider_for_request(self):
        preference = self.provider_preference
        kimi_ready = bool(getattr(self, "kimi", None) and self.kimi.available())

        if preference == "nvidia":
            return "nvidia"
        if preference == "hybrid":
            if kimi_ready:
                return "nvidia"
            ready, _ = self.ollama.available()
            return "ollama" if ready else "nvidia"
        if preference == "gemini":
            return "gemini"
        if preference == "ollama":
            return "ollama"

        # auto: use Kimi when an NVIDIA key is present; otherwise preserve the
        # prior local-first behavior, then Gemini if configured.
        if kimi_ready:
            return "nvidia"
        ready, _ = self.ollama.available()
        if ready:
            return "ollama"
        return "gemini"

    def runtime_label(self):
        preference = self.provider_preference
        kimi_ready = bool(getattr(self, "kimi", None) and self.kimi.available())
        if preference in {"hybrid", "nvidia"}:
            if kimi_ready:
                prefix = "HYBRID" if preference == "hybrid" else "NVIDIA"
                return f"{prefix} · {self.kimi.model}"
            if preference == "nvidia":
                return f"NVIDIA KEY MISSING · {self.kimi.model}"

        ready, _, age = self.ollama.cached_available()
        if age > 5.0:
            self.ollama.refresh_async()
        if ready:
            prefix = "HYBRID FALLBACK" if preference == "hybrid" else "LOCAL"
            return f"{prefix} · {self.ollama.model}"
        if preference == "ollama":
            return f"LOCAL MISSING · {self.ollama.model}"
        if preference == "hybrid":
            return "HYBRID · Kimi unavailable · local model unavailable"
        return f"GEMINI · {self.model}"

    def local_or_fallback_stream(self, prompt, max_tokens, image_bytes_list=None):
        provider = provider_for_request(self)

        if provider == "nvidia":
            self._active_provider = "nvidia"
            try:
                yielded = False
                for piece in self.kimi.chat_stream(
                    prompt,
                    max_tokens=max_tokens,
                    image_bytes_list=image_bytes_list,
                    system_prompt=SYSTEM_PROMPT,
                ):
                    yielded = True
                    yield piece
                if yielded:
                    return
                raise RuntimeError("Kimi-K3 returned no answer tokens")
            except Exception as exc:
                if self.provider_preference == "nvidia":
                    raise
                print(f"[kimi] Cloud request failed; trying local Qwen: {exc}")
                ready, _ = self.ollama.available(cache_seconds=0)
                if ready:
                    self._active_provider = "ollama"
                    yielded = False
                    for piece in self.ollama.chat_stream(
                        prompt,
                        max_tokens=max_tokens,
                        image_bytes_list=image_bytes_list,
                    ):
                        yielded = True
                        yield piece
                    if yielded:
                        return
                if self.provider_preference == "hybrid":
                    raise RuntimeError(
                        f"Kimi-K3 failed ({exc}) and local Qwen fallback is unavailable."
                    ) from exc

        yield from original_local_or_fallback(
            self,
            prompt,
            max_tokens=max_tokens,
            image_bytes_list=image_bytes_list,
        )

    CopilotAI._normalize_provider = staticmethod(normalize_provider)
    CopilotAI.__init__ = patched_init
    CopilotAI._provider_for_request = provider_for_request
    CopilotAI.runtime_label = runtime_label
    CopilotAI._local_or_fallback_stream = local_or_fallback_stream
    CopilotAI._kimi_runtime_patch_installed = True
