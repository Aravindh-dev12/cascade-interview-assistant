import base64
import io
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

import config

load_dotenv()

DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", config.DEFAULT_GEMINI_MODEL)
PROJECT_DIR = Path(__file__).resolve().parent.parent
KB_PATH = PROJECT_DIR / "data" / "interview_knowledge.json"

SYSTEM_PROMPT = (
    "You are a low-latency technical interview practice coach. Give the candidate-ready answer immediately. "
    "Be concise, natural, and decisive. For conceptual questions, start with a 20-40 second spoken answer, "
    "then add only essential supporting points. For MCQs, put the best option on the first line. For coding "
    "questions, detect the requested language, give the approach briefly, then correct code, then time/space "
    "complexity. For debugging, identify the bug and corrected code. Use screenshot details when supplied. "
    "Never invent personal experience, resume facts, metrics, or project details that are not present in the "
    "provided practice context. Do not repeat the question and do not add filler."
)


def _practice_mode_enabled():
    return os.environ.get("PRACTICE_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}


def _load_interview_knowledge():
    try:
        with open(KB_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        print(f"[knowledge] Could not load interview KB: {exc}")
        return {}


class _OllamaClient:
    def __init__(self, base_url, model, num_ctx=8192):
        self.base_url = str(base_url or config.DEFAULT_OLLAMA_BASE_URL).rstrip("/")
        self.model = str(model or config.DEFAULT_LOCAL_MODEL).strip()
        self.num_ctx = max(2048, int(num_ctx or 8192))
        self.keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")
        self._status_cache = (0.0, False, "")
        self._refresh_lock = threading.Lock()

    def reconfigure(self, base_url=None, model=None, num_ctx=None):
        next_base_url = str(base_url or self.base_url).rstrip("/")
        next_model = str(model or self.model).strip()
        next_num_ctx = max(2048, int(num_ctx if num_ctx is not None else self.num_ctx))
        changed = (
            next_base_url != self.base_url
            or next_model != self.model
            or next_num_ctx != self.num_ctx
        )
        self.base_url = next_base_url
        self.model = next_model
        self.num_ctx = next_num_ctx
        if changed:
            self._status_cache = (0.0, False, "")

    def _request(self, path, payload=None, timeout=2.0):
        url = f"{self.base_url}{path}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        return urllib.request.urlopen(request, timeout=timeout)

    def available(self, cache_seconds=2.0):
        now = time.monotonic()
        cached_at, ready, detail = self._status_cache
        if now - cached_at < cache_seconds:
            return ready, detail
        try:
            with self._request("/api/tags", timeout=0.35) as response:
                payload = json.loads(response.read().decode("utf-8"))
            names = [str(item.get("name", "")) for item in payload.get("models", [])]
            ready = self.model in names or any(name.split(":", 1)[0] == self.model for name in names)
            detail = self.model if ready else f"model not installed: {self.model}"
        except Exception as exc:
            ready = False
            detail = str(exc)
        self._status_cache = (now, ready, detail)
        return ready, detail

    def cached_available(self):
        cached_at, ready, detail = self._status_cache
        return ready, detail, max(0.0, time.monotonic() - cached_at) if cached_at else float("inf")

    def refresh_async(self):
        if not self._refresh_lock.acquire(blocking=False):
            return

        def refresh():
            try:
                self.available(cache_seconds=0)
            finally:
                self._refresh_lock.release()

        threading.Thread(target=refresh, daemon=True, name="ollama-status").start()

    def warmup(self):
        ready, _ = self.available(cache_seconds=0)
        if not ready:
            return False
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Reply with one word."},
                {"role": "user", "content": "ready"},
            ],
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": min(self.num_ctx, 4096),
                "num_predict": 1,
                "temperature": 0,
            },
        }
        try:
            with self._request("/api/chat", payload=payload, timeout=120) as response:
                response.read()
            print(f"[ollama] Warmed local model: {self.model}")
            return True
        except Exception as exc:
            print(f"[ollama] Warmup failed: {exc}")
            return False

    def chat_stream(self, prompt, max_tokens, image_bytes_list=None):
        user_message = {"role": "user", "content": prompt}
        if image_bytes_list:
            user_message["images"] = [
                base64.b64encode(image_bytes).decode("ascii")
                for image_bytes in image_bytes_list
                if image_bytes
            ]

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                user_message,
            ],
            "stream": True,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": int(max_tokens),
                "temperature": float(os.environ.get("OLLAMA_TEMPERATURE", "0.15")),
                "top_p": float(os.environ.get("OLLAMA_TOP_P", "0.9")),
            },
        }

        try:
            with self._request("/api/chat", payload=payload, timeout=180) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    message = item.get("message") or {}
                    content = message.get("content") or ""
                    if content:
                        yield content
                    if item.get("done"):
                        break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama unavailable at {self.base_url}: {exc.reason}") from exc


class CopilotAI:
    """Low-latency local-first text/vision engine with Gemini fallback."""

    def __init__(self, model=None, api_key=None, provider=None):
        settings = config.load_settings()
        self.provider_preference = self._normalize_provider(
            provider or os.environ.get("AI_PROVIDER") or settings.get("ai_provider", "auto")
        )
        self.model = self._normalize_model(model or settings.get("model"))
        self.api_key = self._resolve_key(api_key)
        self._gemini_client = None
        self.interview_knowledge = _load_interview_knowledge()
        self.transcript_history = []
        self.max_transcript_history = max(4, int(os.environ.get("TRANSCRIPT_CONTEXT_LINES", "12")))
        self.image_history = []
        self.image_fingerprints = []
        self.max_image_history = max(1, int(os.environ.get("PRACTICE_IMAGE_CONTEXT_FRAMES", "3")))

        local_model = os.environ.get("OLLAMA_MODEL", "").strip() or settings.get(
            "local_model", config.DEFAULT_LOCAL_MODEL
        )
        base_url = os.environ.get("OLLAMA_BASE_URL", "").strip() or settings.get(
            "ollama_base_url", config.DEFAULT_OLLAMA_BASE_URL
        )
        num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", settings.get("ollama_num_ctx", 8192)))
        self.ollama = _OllamaClient(base_url, local_model, num_ctx=num_ctx)
        self._active_provider = "gemini"

        if self.provider_preference in {"auto", "ollama"}:
            threading.Thread(target=self.ollama.warmup, daemon=True, name="ollama-warmup").start()

    @staticmethod
    def _normalize_provider(provider):
        provider = str(provider or "auto").strip().lower()
        return provider if provider in {"auto", "ollama", "gemini"} else "auto"

    @staticmethod
    def _normalize_model(model):
        model = str(model or "").strip()
        return model if model.startswith("gemini-") else DEFAULT_GEMINI_MODEL

    @staticmethod
    def _resolve_key(api_key=None):
        supplied = (api_key or "").strip()
        if supplied:
            return supplied
        return os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()

    def _reload_local_settings(self):
        settings = config.load_settings()
        self.provider_preference = self._normalize_provider(
            os.environ.get("AI_PROVIDER", "").strip() or settings.get("ai_provider", self.provider_preference)
        )
        local_model = os.environ.get("OLLAMA_MODEL", "").strip() or settings.get(
            "local_model", config.DEFAULT_LOCAL_MODEL
        )
        base_url = os.environ.get("OLLAMA_BASE_URL", "").strip() or settings.get(
            "ollama_base_url", config.DEFAULT_OLLAMA_BASE_URL
        )
        num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", settings.get("ollama_num_ctx", 8192)))
        self.ollama.reconfigure(base_url=base_url, model=local_model, num_ctx=num_ctx)

    def set_config(self, model=None, api_key=None, provider=None):
        next_model = self._normalize_model(model)
        next_key = self._resolve_key(api_key)
        if next_key != self.api_key:
            self._gemini_client = None
        self.model = next_model
        self.api_key = next_key
        if provider is not None:
            self.provider_preference = self._normalize_provider(provider)
        self._reload_local_settings()

    def runtime_label(self):
        if self.provider_preference in {"auto", "ollama"}:
            ready, _, age = self.ollama.cached_available()
            if age > 5.0:
                self.ollama.refresh_async()
            if ready:
                return f"LOCAL · {self.ollama.model}"
            if self.provider_preference == "ollama":
                return f"LOCAL MISSING · {self.ollama.model}"
            if age == float("inf"):
                return f"AUTO · {self.ollama.model}"
        return f"GEMINI · {self.model}"

    def add_transcript_line(self, speaker, text):
        text = str(text or "").strip()
        if not text:
            return
        self.transcript_history.append({"speaker": speaker, "text": text})
        if len(self.transcript_history) > self.max_transcript_history:
            del self.transcript_history[:-self.max_transcript_history]

    def clear_history(self):
        self.transcript_history.clear()
        self.image_history.clear()
        self.image_fingerprints.clear()

    def get_formatted_transcript(self):
        if not self.transcript_history:
            return "[No conversation recorded yet]"
        return "\n".join(f"{item['speaker']}: {item['text']}" for item in self.transcript_history)

    def _knowledge_context(self):
        if not _practice_mode_enabled() or not self.interview_knowledge:
            return ""
        return json.dumps(self.interview_knowledge, ensure_ascii=False, separators=(",", ":"))

    def _build_prompt(self, custom_query=None, image_task=None):
        task = custom_query or "Answer the latest substantive interviewer question in the practice transcript."
        parts = []
        knowledge = self._knowledge_context()
        if knowledge:
            parts.append(
                "Candidate/role practice context follows. Use only when relevant; never invent beyond it:\n" + knowledge
            )
        parts.append("Recent transcript:\n" + self.get_formatted_transcript())
        if image_task:
            parts.append("Screen context instructions:\n" + image_task)
        parts.append("Current task:\n" + task)
        return "\n\n".join(parts)

    @staticmethod
    def _image_fingerprint(image_bytes):
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("L").resize((32, 32))
        return image.tobytes()

    @staticmethod
    def _fingerprint_distance(left, right):
        if not left or not right or len(left) != len(right):
            return 1.0
        return sum(abs(a - b) for a, b in zip(left, right)) / (255.0 * len(left))

    def _remember_image(self, image_bytes):
        try:
            fp = self._image_fingerprint(image_bytes)
        except Exception:
            fp = b""
        duplicate_threshold = float(os.environ.get("PRACTICE_IMAGE_DUPLICATE_THRESHOLD", "0.012"))
        if self.image_fingerprints and fp:
            if self._fingerprint_distance(self.image_fingerprints[-1], fp) <= duplicate_threshold:
                self.image_history[-1] = image_bytes
                self.image_fingerprints[-1] = fp
                return
        self.image_history.append(image_bytes)
        self.image_fingerprints.append(fp)
        if len(self.image_history) > self.max_image_history:
            del self.image_history[:-self.max_image_history]
            del self.image_fingerprints[:-self.max_image_history]

    def _get_gemini_client(self):
        if not self.api_key:
            self.api_key = self._resolve_key()
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured in the project env file")
        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=self.api_key)
        return self._gemini_client

    def _gemini_stream(self, contents, max_tokens=None):
        thinking_budget = int(os.environ.get("GEMINI_THINKING_BUDGET", "0"))
        response = self._get_gemini_client().models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=max_tokens or int(os.environ.get("GEMINI_TEXT_MAX_TOKENS", "700")),
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            ),
        )
        for chunk in response:
            text = getattr(chunk, "text", None)
            if text:
                yield text

    def _provider_for_request(self):
        if self.provider_preference == "gemini":
            return "gemini"
        ready, _ = self.ollama.available()
        if ready:
            return "ollama"
        if self.provider_preference == "ollama":
            return "ollama"
        return "gemini"

    def _local_or_fallback_stream(self, prompt, max_tokens, image_bytes_list=None):
        provider = self._provider_for_request()
        if provider == "ollama":
            self._active_provider = "ollama"
            try:
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
            except Exception as exc:
                if self.provider_preference == "ollama" or not self.api_key:
                    raise RuntimeError(
                        f"Local model failed: {exc}. Install/start Ollama and run `ollama pull {self.ollama.model}`."
                    ) from exc
                print(f"[ollama] Local request failed; falling back to Gemini: {exc}")

        self._active_provider = "gemini"
        if image_bytes_list:
            from PIL import Image

            contents = [prompt]
            for frame in image_bytes_list:
                contents.append(Image.open(io.BytesIO(frame)).convert("RGB"))
            yield from self._gemini_stream(contents, max_tokens=max_tokens)
        else:
            yield from self._gemini_stream(prompt, max_tokens=max_tokens)

    def generate_text_stream(self, custom_query=None):
        prompt = self._build_prompt(custom_query)
        max_tokens = int(os.environ.get("LOCAL_TEXT_MAX_TOKENS", os.environ.get("GEMINI_TEXT_MAX_TOKENS", "700")))
        yield from self._local_or_fallback_stream(prompt, max_tokens=max_tokens)

    def generate_vision_stream(self, image_bytes, custom_query=None, use_image_history=False):
        if use_image_history:
            self._remember_image(image_bytes)
            frames = list(self.image_history)
        else:
            frames = [image_bytes]

        image_task = (
            "Read visible text/code carefully. Determine whether this is a coding problem, MCQ, debugging task, "
            "terminal error, diagram, system-design prompt, or conceptual question. Use all supplied frames as one "
            "chronological problem context when multiple frames are present. Deduplicate overlap caused by scrolling. "
            "Do not guess missing constraints."
        )
        prompt = self._build_prompt(
            custom_query or "Solve or explain the current visible practice question. Give the useful answer first.",
            image_task=image_task,
        )
        max_tokens = int(os.environ.get("LOCAL_VISION_MAX_TOKENS", os.environ.get("GEMINI_VISION_MAX_TOKENS", "1600")))
        yield from self._local_or_fallback_stream(prompt, max_tokens=max_tokens, image_bytes_list=frames)

    def generate_answer_stream(self, image_bytes=None, custom_query=None, use_image_history=False):
        if not _practice_mode_enabled():
            yield (
                "### Practice mode is off\n\n"
                "Set `PRACTICE_MODE=1` only for mock interviews, practice, or sessions where AI assistance is explicitly permitted."
            )
            return
        try:
            if image_bytes is not None:
                yield from self.generate_vision_stream(
                    image_bytes,
                    custom_query=custom_query,
                    use_image_history=use_image_history,
                )
            else:
                yield from self.generate_text_stream(custom_query)
        except Exception as exc:
            yield f"### AI Error\n\n`{exc}`"

    def generate_answer(self, image_bytes=None, custom_query=None, use_image_history=False):
        return "".join(
            self.generate_answer_stream(
                image_bytes=image_bytes,
                custom_query=custom_query,
                use_image_history=use_image_history,
            )
        ).strip()
