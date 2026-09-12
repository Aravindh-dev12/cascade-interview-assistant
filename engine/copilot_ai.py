import base64
import io
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import config
from engine.nvidia_kimi import NvidiaKimiClient

PROJECT_DIR = Path(__file__).resolve().parent.parent
KB_PATH = PROJECT_DIR / "data" / "interview_knowledge.json"

SYSTEM_PROMPT = (
    "You are a low-latency technical interview practice coach. Give the candidate-ready answer immediately. "
    "Be concise, natural, and decisive. For conceptual questions, start with a 20-40 second spoken answer, "
    "then add only essential supporting points. For MCQs, put the best option on the first line. For coding "
    "questions, detect the requested language, give the approach briefly, then correct code, then time/space "
    "complexity. For debugging, identify the bug and corrected code. Use screenshot/camera details when supplied. "
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
        age = max(0.0, time.monotonic() - cached_at) if cached_at else float("inf")
        return ready, detail, age

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
    """NVIDIA Kimi-K3 + local Qwen text/vision engine."""

    def __init__(self, provider=None):
        settings = config.load_settings()
        self.provider_preference = self._normalize_provider(
            provider or os.environ.get("AI_PROVIDER") or settings.get("ai_provider", "hybrid")
        )
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
        self.kimi = NvidiaKimiClient(api_key=os.environ.get("NVIDIA_API_KEY", ""))
        self._active_provider = "nvidia" if self.kimi.available() else "ollama"

        if self.provider_preference in {"hybrid", "ollama"}:
            threading.Thread(target=self.ollama.warmup, daemon=True, name="ollama-warmup").start()

    @staticmethod
    def _normalize_provider(provider):
        provider = str(provider or "hybrid").strip().lower()
        return provider if provider in {"hybrid", "nvidia", "ollama"} else "hybrid"

    def _reload_settings(self):
        settings = config.load_settings()
        self.provider_preference = self._normalize_provider(
            os.environ.get("AI_PROVIDER", "").strip()
            or settings.get("ai_provider", self.provider_preference)
        )
        local_model = os.environ.get("OLLAMA_MODEL", "").strip() or settings.get(
            "local_model", config.DEFAULT_LOCAL_MODEL
        )
        base_url = os.environ.get("OLLAMA_BASE_URL", "").strip() or settings.get(
            "ollama_base_url", config.DEFAULT_OLLAMA_BASE_URL
        )
        num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", settings.get("ollama_num_ctx", 8192)))
        self.ollama.reconfigure(base_url=base_url, model=local_model, num_ctx=num_ctx)
        self.kimi.reconfigure(api_key=os.environ.get("NVIDIA_API_KEY", "").strip())

    def set_config(self, provider=None):
        if provider is not None:
            self.provider_preference = self._normalize_provider(provider)
        self._reload_settings()

    def runtime_label(self):
        ready, _, age = self.ollama.cached_available()
        if age > 5.0:
            self.ollama.refresh_async()
        if self.provider_preference == "nvidia":
            return f"NVIDIA · {self.kimi.model}" if self.kimi.available() else "NVIDIA KEY MISSING"
        if self.provider_preference == "ollama":
            return f"LOCAL · {self.ollama.model}" if ready else f"LOCAL MISSING · {self.ollama.model}"
        if self.kimi.available():
            fallback = self.ollama.model if ready else "Qwen fallback"
            return f"HYBRID · Kimi K3 + {fallback}"
        return f"HYBRID LOCAL · {self.ollama.model}" if ready else "HYBRID · no available model"

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
            parts.append("Visual context instructions:\n" + image_task)
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

    def _qwen_stream(self, prompt, max_tokens, image_bytes_list=None):
        ready, detail = self.ollama.available(cache_seconds=0)
        if not ready:
            raise RuntimeError(
                f"Local Qwen unavailable: {detail}. Run `ollama pull {self.ollama.model}` and start Ollama."
            )
        self._active_provider = "ollama"
        yield from self.ollama.chat_stream(
            prompt,
            max_tokens=max_tokens,
            image_bytes_list=image_bytes_list,
        )

    def _kimi_stream(self, prompt, max_tokens, image_bytes_list=None):
        self.kimi.reconfigure(api_key=os.environ.get("NVIDIA_API_KEY", "").strip())
        if not self.kimi.available():
            raise RuntimeError("NVIDIA_API_KEY is missing from the project .env file")
        self._active_provider = "nvidia"
        yield from self.kimi.chat_stream(
            prompt,
            max_tokens=max_tokens,
            image_bytes_list=image_bytes_list,
            system_prompt=SYSTEM_PROMPT,
        )

    @staticmethod
    def _run_provider(name, factory, events):
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

    def _hybrid_stream(self, prompt, max_tokens, image_bytes_list=None):
        if not self.kimi.available():
            yield from self._qwen_stream(prompt, max_tokens, image_bytes_list)
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
            target=self._run_provider,
            args=("kimi", lambda: self._kimi_stream(prompt, max_tokens, image_bytes_list), events),
            daemon=True,
            name="hybrid-kimi",
        ).start()

        while winner is None:
            elapsed = time.monotonic() - started
            if not qwen_started and elapsed >= hedge_seconds:
                qwen_started = True
                print("[hybrid] Kimi has no first token yet; starting local Qwen hedge.")
                threading.Thread(
                    target=self._run_provider,
                    args=("qwen", lambda: self._qwen_stream(prompt, max_tokens, image_bytes_list), events),
                    daemon=True,
                    name="hybrid-qwen",
                ).start()

            try:
                name, event_type, payload = events.get(timeout=0.05)
            except queue.Empty:
                continue

            if event_type == "chunk":
                winner = name
                self._active_provider = "nvidia" if name == "kimi" else "ollama"
                print(f"[hybrid] {name.upper()} won first token in {time.monotonic() - started:.2f}s")
                yield payload
                break
            if event_type == "error":
                errors[name] = payload
                print(f"[hybrid] {name} failed before first token: {payload}")
                if name == "kimi" and not qwen_started:
                    qwen_started = True
                    threading.Thread(
                        target=self._run_provider,
                        args=("qwen", lambda: self._qwen_stream(prompt, max_tokens, image_bytes_list), events),
                        daemon=True,
                        name="hybrid-qwen",
                    ).start()
            elif event_type == "done":
                done.add(name)
                if payload:
                    errors[name] = RuntimeError(str(payload))

            if qwen_started and len(errors) + len(done) >= 2 and winner is None:
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

    def _route_stream(self, prompt, max_tokens, image_bytes_list=None):
        self._reload_settings()
        if self.provider_preference == "nvidia":
            yield from self._kimi_stream(prompt, max_tokens, image_bytes_list)
            return
        if self.provider_preference == "ollama":
            yield from self._qwen_stream(prompt, max_tokens, image_bytes_list)
            return
        yield from self._hybrid_stream(prompt, max_tokens, image_bytes_list)

    def generate_text_stream(self, custom_query=None):
        prompt = self._build_prompt(custom_query)
        max_tokens = int(os.environ.get("TEXT_MAX_TOKENS", "700"))
        yield from self._route_stream(prompt, max_tokens=max_tokens)

    def generate_vision_stream(self, image_bytes, custom_query=None, use_image_history=False):
        if use_image_history:
            self._remember_image(image_bytes)
            frames = list(self.image_history)
        else:
            frames = [image_bytes]

        image_task = (
            "Read visible text/code carefully. Determine whether this is a coding problem, MCQ, debugging task, "
            "terminal error, diagram, system-design prompt, conceptual question, document, or camera object. "
            "Use all supplied frames as one chronological problem context when multiple frames are present. "
            "Deduplicate overlap caused by scrolling. Do not guess missing constraints."
        )
        prompt = self._build_prompt(
            custom_query or "Solve or explain the current visible practice question. Give the useful answer first.",
            image_task=image_task,
        )
        max_tokens = int(os.environ.get("VISION_MAX_TOKENS", "1200"))
        yield from self._route_stream(prompt, max_tokens=max_tokens, image_bytes_list=frames)

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
