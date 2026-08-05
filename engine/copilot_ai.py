import io
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


TEXT_MODEL = os.environ.get("NVIDIA_TEXT_MODEL", "z-ai/glm-5.2")
NVIDIA_BASE_URL = os.environ.get(
    "NVIDIA_INTEGRATE_BASE_URL", "https://integrate.api.nvidia.com/v1"
)

SYSTEM_PROMPT = (
    "You are a concise technical interview practice coach. Answer the latest question "
    "directly. Put the most useful answer first. For coding questions, give the core "
    "approach and compact code, then time and space complexity. Avoid filler."
)


def _practice_mode_enabled():
    return os.environ.get("PRACTICE_MODE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


class CopilotAI:
    """Text coaching via NVIDIA GLM-5.2, with Gemini retained only for vision."""

    def __init__(self, provider="nvidia", model=TEXT_MODEL, api_key=None):
        self.provider = "nvidia"
        self.model = TEXT_MODEL
        self.api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
        self._nvidia_client = None

        self.transcript_history = []
        self.max_transcript_history = 12

    def set_config(self, provider=None, model=None, api_key=None):
        self.provider = "nvidia"
        self.model = TEXT_MODEL
        refreshed_key = os.environ.get("NVIDIA_API_KEY", "").strip()
        if refreshed_key != self.api_key:
            self._nvidia_client = None
        self.api_key = refreshed_key

    def add_transcript_line(self, speaker, text):
        self.transcript_history.append({"speaker": speaker, "text": text})
        if len(self.transcript_history) > self.max_transcript_history:
            self.transcript_history.pop(0)

    def clear_history(self):
        self.transcript_history.clear()

    def get_formatted_transcript(self):
        if not self.transcript_history:
            return "[No conversation recorded yet]"
        return "\n".join(
            f"{item['speaker']}: {item['text']}" for item in self.transcript_history
        )

    def _get_nvidia_client(self):
        if not self.api_key:
            self.api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY is not configured")
        if self._nvidia_client is None:
            self._nvidia_client = OpenAI(
                base_url=NVIDIA_BASE_URL,
                api_key=self.api_key,
                timeout=15.0,
                max_retries=0,
            )
        return self._nvidia_client

    def _build_text_messages(self, custom_query=None):
        transcript = self.get_formatted_transcript()
        if custom_query:
            task = custom_query
        else:
            task = "Answer the latest substantive question in this practice transcript."

        prompt = f"Transcript:\n{transcript}\n\nTask:\n{task}"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    def stream_text_answer(self, custom_query=None):
        """Yield GLM-5.2 chunks as soon as NVIDIA returns them."""
        client = self._get_nvidia_client()
        stream = client.chat.completions.create(
            model=self.model,
            messages=self._build_text_messages(custom_query),
            temperature=0.2,
            top_p=0.9,
            max_tokens=int(os.environ.get("NVIDIA_TEXT_MAX_TOKENS", "256")),
            stream=True,
        )

        for chunk in stream:
            if not getattr(chunk, "choices", None) or not chunk.choices:
                continue
            delta = getattr(chunk.choices[0], "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content:
                yield content

    def generate_text_answer(self, custom_query=None):
        """Return a useful compact answer quickly instead of waiting for a long stream."""
        pieces = []
        total_chars = 0
        min_chars = int(os.environ.get("NVIDIA_FAST_MIN_CHARS", "220"))
        hard_chars = int(os.environ.get("NVIDIA_FAST_MAX_CHARS", "900"))

        for piece in self.stream_text_answer(custom_query):
            pieces.append(piece)
            total_chars += len(piece)
            joined = "".join(pieces)

            # Once we have a useful amount of text, return on a natural boundary.
            if total_chars >= min_chars and joined.rstrip().endswith((".", "!", "?", "```")):
                break
            if total_chars >= hard_chars:
                break

        return "".join(pieces).strip()

    def generate_answer(self, image_bytes=None, custom_query=None):
        """Compatibility method used by the current UI worker."""
        if image_bytes is not None:
            return self._call_gemini_vision(image_bytes, custom_query)

        if custom_query is None and not _practice_mode_enabled():
            return (
                "### Live transcription active\n\n"
                "Automatic coaching is disabled. Set `PRACTICE_MODE=1` only for "
                "mock interviews or sessions where AI assistance is explicitly permitted."
            )

        try:
            return self.generate_text_answer(custom_query)
        except Exception as exc:
            return f"### NVIDIA GLM-5.2 Error\n\n`{exc}`"

    def _call_gemini_vision(self, image_bytes, custom_query=None):
        """GLM-5.2 is text-only; retain the existing Gemini vision path."""
        if genai is None:
            return "### Vision unavailable\n\nInstall `google-genai` to use screen-image analysis."

        gemini_key = os.environ.get(
            "GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", "")
        ).strip()
        if not gemini_key:
            return (
                "### Vision key missing\n\n"
                "GLM-5.2 handles text answers. Set `GEMINI_API_KEY` only if you also "
                "want the existing image-analysis feature."
            )

        transcript = self.get_formatted_transcript()
        task = custom_query or "Explain the problem shown in this image for practice."
        prompt = f"Transcript:\n{transcript}\n\nTask:\n{task}"

        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes))
        client = genai.Client(api_key=gemini_key)
        response = client.models.generate_content(
            model=os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-flash"),
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1024,
            ),
        )
        return response.text or ""
