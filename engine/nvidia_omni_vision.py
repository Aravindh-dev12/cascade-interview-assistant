import base64
import json
import os
import time
import urllib.error
import urllib.request


DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"


class NvidiaOmniVisionClient:
    """NVIDIA Nemotron Omni image-to-text client.

    The NVIDIA API key is read only from the environment. This client turns a
    manually captured screen/camera frame into text context; local Qwen produces
    the final answer. Transient hosted-worker saturation is retried before the
    caller falls back to local Qwen vision.
    """

    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = (api_key or os.environ.get("NVIDIA_API_KEY", "")).strip()
        self.base_url = str(
            base_url or os.environ.get("NVIDIA_VISION_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        self.model = str(
            model or os.environ.get("NVIDIA_VISION_MODEL", DEFAULT_MODEL)
        ).strip()
        self._blocked_until = 0.0

    def reconfigure(self):
        self.api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
        self.base_url = os.environ.get(
            "NVIDIA_VISION_BASE_URL", self.base_url or DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = os.environ.get(
            "NVIDIA_VISION_MODEL", self.model or DEFAULT_MODEL
        ).strip()

    def available(self):
        return bool(self.api_key and self.base_url and self.model)

    @staticmethod
    def _data_url(image_bytes):
        mime = "image/png" if image_bytes[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    @staticmethod
    def _extract_text(payload):
        if not isinstance(payload, dict):
            return ""
        choices = payload.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content") or ""
                        if text:
                            parts.append(str(text))
                return "\n".join(parts).strip()
        for key in ("output_text", "text", "result"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "quntumnintent/1.0",
        }

    def _open_json(self, request, timeout):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = int(getattr(response, "status", response.getcode()))
                raw = response.read().decode("utf-8", errors="replace")
                payload = json.loads(raw) if raw.strip() else {}
                return status, payload
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
            if exc.code == 429:
                backoff = max(
                    10.0,
                    float(os.environ.get("NVIDIA_VISION_429_BACKOFF_SECONDS", "60")),
                )
                self._blocked_until = time.monotonic() + backoff
                raise RuntimeError(
                    f"NVIDIA Omni vision rate-limited (HTTP 429); pausing cloud vision for {backoff:.0f}s"
                ) from exc
            raise RuntimeError(f"NVIDIA Omni vision HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"NVIDIA Omni vision unavailable: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("NVIDIA Omni vision request timed out") from exc

    @staticmethod
    def _is_transient_overload(exc):
        text = str(exc).lower()
        return (
            "http 503" in text
            or "resourceexhausted" in text
            or "service unavailable" in text
            or "worker local total request limit reached" in text
        )

    def _open_json_with_retry(self, request, timeout, purpose="request"):
        retries = max(0, min(6, int(os.environ.get("NVIDIA_VISION_503_RETRIES", "3"))))
        base_delay = max(
            0.2,
            min(5.0, float(os.environ.get("NVIDIA_VISION_503_RETRY_DELAY_SECONDS", "0.8"))),
        )
        for attempt in range(retries + 1):
            try:
                return self._open_json(request, timeout=timeout)
            except Exception as exc:
                if not self._is_transient_overload(exc) or attempt >= retries:
                    raise
                delay = min(5.0, base_delay * (2 ** attempt))
                print(
                    f"[vision-nvidia] Hosted worker busy during {purpose}; "
                    f"retry {attempt + 1}/{retries} in {delay:.1f}s"
                )
                time.sleep(delay)
        raise RuntimeError("NVIDIA Omni vision retry loop exhausted")

    def _poll(self, request_id, deadline):
        poll_seconds = max(
            0.2,
            float(os.environ.get("NVIDIA_VISION_POLL_INTERVAL_SECONDS", "0.5")),
        )
        while time.monotonic() < deadline:
            request = urllib.request.Request(
                f"{self.base_url}/status/{request_id}",
                headers=self._headers(),
                method="GET",
            )
            remaining = max(1.0, deadline - time.monotonic())
            status, payload = self._open_json_with_retry(
                request,
                timeout=min(10.0, remaining),
                purpose="polling",
            )
            if status == 200:
                return payload
            if status != 202:
                raise RuntimeError(f"Unexpected NVIDIA vision polling status: {status}")
            time.sleep(min(poll_seconds, max(0.05, deadline - time.monotonic())))
        raise RuntimeError("NVIDIA Omni vision polling timed out")

    def extract(self, image_bytes, task_hint=""):
        self.reconfigure()
        if not self.available():
            raise RuntimeError("NVIDIA_API_KEY is missing; image-to-text is unavailable")
        if time.monotonic() < self._blocked_until:
            remaining = self._blocked_until - time.monotonic()
            raise RuntimeError(
                f"NVIDIA Omni vision is temporarily paused after rate limiting ({remaining:.0f}s remaining)"
            )
        if not image_bytes:
            raise RuntimeError("No image bytes supplied to NVIDIA Omni vision")

        prompt = (
            "Transcribe this interview-practice screenshot accurately for a second AI that will solve it. "
            "Do NOT answer with only a category such as 'coding', 'MCQ', or 'general'. The actual visible "
            "question text is mandatory whenever readable. Read the entire visible question, every answer "
            "option, code, function signature, starter code, terminal/output text, constraints, examples, "
            "tables, labels, and requested programming language. Preserve identifiers, operators, numbers, "
            "and code exactly when readable. Do not invent text hidden by another window or outside the image. "
            "Return this compact structure:\n"
            "TYPE: <MCQ/coding/debugging/math/SQL/output/system-design/general>\n"
            "QUESTION: <actual visible question/problem statement>\n"
            "OPTIONS: <all visible options, or NONE>\n"
            "CODE_OR_CONTEXT: <visible code/constraints/examples/terminal text, or NONE>\n"
            "If there is genuinely no readable question, return NO_READABLE_QUESTION followed by a short "
            "description of what text is visible. Do not solve the problem."
        )
        if task_hint:
            prompt += f"\nUser intent: {task_hint}"

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": self._data_url(image_bytes)},
                        },
                    ],
                }
            ],
            "model": self.model,
            "max_tokens": max(
                256,
                int(os.environ.get("NVIDIA_VISION_MAX_TOKENS", "1400")),
            ),
            "stream": False,
            "temperature": float(os.environ.get("NVIDIA_VISION_TEMPERATURE", "0.2")),
            "top_k": max(1, int(os.environ.get("NVIDIA_VISION_TOP_K", "1"))),
            "chat_template_kwargs": {"enable_thinking": False},
        }

        timeout = max(
            8.0,
            float(os.environ.get("NVIDIA_VISION_TIMEOUT_SECONDS", "30")),
        )
        poll_timeout = max(
            timeout,
            float(os.environ.get("NVIDIA_VISION_POLL_TIMEOUT_SECONDS", "40")),
        )
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        started = time.monotonic()
        status, response_payload = self._open_json_with_retry(
            request,
            timeout=timeout,
            purpose="image extraction",
        )
        if status == 202:
            request_id = (
                response_payload.get("requestId")
                or response_payload.get("request_id")
                or response_payload.get("id")
            )
            if not request_id:
                raise RuntimeError("NVIDIA Omni returned 202 without a requestId")
            response_payload = self._poll(
                str(request_id),
                deadline=started + poll_timeout,
            )
        elif status != 200:
            raise RuntimeError(f"Unexpected NVIDIA Omni response status: {status}")

        text = self._extract_text(response_payload)
        if not text:
            raise RuntimeError("NVIDIA Omni vision returned no image text")
        print(
            f"[vision-nvidia] Extracted visual context in {time.monotonic() - started:.2f}s · "
            f"chars={len(text)}"
        )
        return text
