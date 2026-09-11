import base64
import json
import os
import urllib.error
import urllib.request


DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "moonshotai/kimi-k3"


class NvidiaKimiClient:
    """Small OpenAI-compatible streaming client for NVIDIA-hosted Kimi K3.

    NVIDIA_API_KEY is intentionally read only from the environment. Never persist
    API keys in settings JSON or source control.
    """

    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = (api_key or os.environ.get("NVIDIA_API_KEY", "")).strip()
        self.base_url = str(
            base_url or os.environ.get("NVIDIA_KIMI_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        self.model = str(
            model or os.environ.get("NVIDIA_KIMI_MODEL", DEFAULT_MODEL)
        ).strip()

    def reconfigure(self, api_key=None, base_url=None, model=None):
        if api_key is not None:
            self.api_key = str(api_key or "").strip()
        else:
            self.api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
        if base_url is not None:
            self.base_url = str(base_url or DEFAULT_BASE_URL).rstrip("/")
        else:
            self.base_url = str(
                os.environ.get("NVIDIA_KIMI_BASE_URL", self.base_url or DEFAULT_BASE_URL)
            ).rstrip("/")
        if model is not None:
            self.model = str(model or DEFAULT_MODEL).strip()
        else:
            self.model = str(
                os.environ.get("NVIDIA_KIMI_MODEL", self.model or DEFAULT_MODEL)
            ).strip()

    def available(self):
        return bool(self.api_key and self.model and self.base_url)

    @staticmethod
    def _data_url(image_bytes):
        mime = "image/png" if image_bytes[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    @staticmethod
    def _extract_delta(item):
        try:
            choices = item.get("choices") or []
            if not choices:
                return ""
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                pieces = []
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text") or part.get("content") or ""
                        if text:
                            pieces.append(str(text))
                return "".join(pieces)
        except Exception:
            pass
        return ""

    def chat_stream(self, prompt, max_tokens=900, image_bytes_list=None, system_prompt=None):
        if not self.api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY is not configured. Rotate the exposed key and place the new key in .env."
            )

        user_content = [{"type": "text", "text": prompt}]
        for image_bytes in image_bytes_list or []:
            if image_bytes:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._data_url(image_bytes)},
                    }
                )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        requested_tokens = int(max_tokens or 900)
        token_cap = max(128, int(os.environ.get("NVIDIA_KIMI_MAX_TOKENS", "1200")))
        payload = {
            "messages": messages,
            "model": self.model,
            "max_tokens": min(requested_tokens, token_cap),
            "seed": int(os.environ.get("NVIDIA_KIMI_SEED", "0")),
            "stream": True,
            "temperature": float(os.environ.get("NVIDIA_KIMI_TEMPERATURE", "1.0")),
        }

        reasoning_effort = os.environ.get(
            "NVIDIA_KIMI_REASONING_EFFORT", "low"
        ).strip().lower()
        if reasoning_effort in {"low", "high", "max"}:
            payload["reasoning_effort"] = reasoning_effort

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
                "User-Agent": "quntumnintent/1.0",
            },
            method="POST",
        )
        timeout = max(
            1.0, float(os.environ.get("NVIDIA_KIMI_TIMEOUT_SECONDS", "2.5"))
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if not line or line == "[DONE]":
                        if line == "[DONE]":
                            break
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = self._extract_delta(item)
                    if text:
                        yield text
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
            raise RuntimeError(f"NVIDIA Kimi HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"NVIDIA Kimi unavailable: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("NVIDIA Kimi timed out before a response arrived") from exc
