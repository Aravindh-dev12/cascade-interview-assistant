import io
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

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
    """Gemini-only answer engine. NVIDIA is used only by the ASR worker."""

    def __init__(self, provider="gemini", model=DEFAULT_GEMINI_MODEL, api_key=None):
        self.provider = "gemini"
        self.model = model if str(model).startswith("gemini-") else DEFAULT_GEMINI_MODEL
        self.api_key = (
            (api_key or "").strip()
            or os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        )
        self._client = None
        self.transcript_history = []
        self.max_transcript_history = 12

    def set_config(self, provider=None, model=None, api_key=None):
        self.provider = "gemini"
        if model and str(model).startswith("gemini-"):
            self.model = model
        else:
            self.model = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

        refreshed_key = (
            (api_key or "").strip()
            or os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        )
        if refreshed_key != self.api_key:
            self._client = None
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

    def _get_client(self):
        if not self.api_key:
            self.api_key = (
                os.environ.get("GEMINI_API_KEY", "").strip()
                or os.environ.get("GOOGLE_API_KEY", "").strip()
            )
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _build_prompt(self, custom_query=None):
        transcript = self.get_formatted_transcript()
        task = custom_query or "Answer the latest substantive question in this practice transcript."
        return f"Transcript:\n{transcript}\n\nTask:\n{task}"

    def _stream(self, contents, max_tokens=None):
        response = self._get_client().models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=max_tokens
                or int(os.environ.get("GEMINI_TEXT_MAX_TOKENS", "384")),
            ),
        )
        for chunk in response:
            text = getattr(chunk, "text", None)
            if text:
                yield text

    def _collect_fast(self, stream):
        pieces = []
        total_chars = 0
        min_chars = int(os.environ.get("GEMINI_FAST_MIN_CHARS", "120"))
        hard_chars = int(os.environ.get("GEMINI_FAST_MAX_CHARS", "700"))

        for piece in stream:
            pieces.append(piece)
            total_chars += len(piece)
            joined = "".join(pieces)
            if total_chars >= min_chars and joined.rstrip().endswith((".", "!", "?", "```")):
                break
            if total_chars >= hard_chars:
                break
        return "".join(pieces).strip()

    def generate_text_answer(self, custom_query=None):
        return self._collect_fast(self._stream(self._build_prompt(custom_query)))

    def generate_vision_answer(self, image_bytes, custom_query=None):
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes))
        prompt = self._build_prompt(
            custom_query or "Explain the problem shown in this image for practice."
        )
        return self._collect_fast(
            self._stream(
                [prompt, image],
                max_tokens=int(os.environ.get("GEMINI_VISION_MAX_TOKENS", "768")),
            )
        )

    def generate_answer(self, image_bytes=None, custom_query=None):
        if custom_query is None and not _practice_mode_enabled():
            return (
                "### Live transcription active\n\n"
                "Automatic coaching is disabled. Set `PRACTICE_MODE=1` only for "
                "mock interviews or sessions where AI assistance is explicitly permitted."
            )

        try:
            if image_bytes is not None:
                return self.generate_vision_answer(image_bytes, custom_query)
            return self.generate_text_answer(custom_query)
        except Exception as exc:
            return f"### Gemini Error\n\n`{exc}`"
