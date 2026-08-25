import io
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")

SYSTEM_PROMPT = (
    "You are a concise technical interview practice coach. Answer the latest question "
    "directly. Put the most useful answer first. For coding questions, give the core "
    "approach and compact code, then time and space complexity. For MCQs, state the "
    "best option first and briefly justify it. Avoid filler."
)


def _practice_mode_enabled():
    return os.environ.get("PRACTICE_MODE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class CopilotAI:
    """Gemini-only answer engine for text and screen/vision prompts."""

    def __init__(self, model=None, api_key=None):
        self.model = self._normalize_model(model)
        self.api_key = self._resolve_key(api_key)
        self._gemini_client = None
        self.transcript_history = []
        self.max_transcript_history = 12

    @staticmethod
    def _normalize_model(model):
        model = str(model or "").strip()
        return model if model.startswith("gemini-") else DEFAULT_GEMINI_MODEL

    @staticmethod
    def _resolve_key(api_key=None):
        supplied = (api_key or "").strip()
        if supplied:
            return supplied
        return (
            os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        )

    def set_config(self, model=None, api_key=None):
        next_model = self._normalize_model(model)
        next_key = self._resolve_key(api_key)
        if next_key != self.api_key:
            self._gemini_client = None
        self.model = next_model
        self.api_key = next_key

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

    def _build_prompt(self, custom_query=None):
        transcript = self.get_formatted_transcript()
        task = custom_query or "Answer the latest substantive question in this practice transcript."
        return f"Transcript:\n{transcript}\n\nTask:\n{task}"

    def _get_gemini_client(self):
        if not self.api_key:
            self.api_key = self._resolve_key()
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=self.api_key)
        return self._gemini_client

    def _collect_fast(self, stream, min_env, max_env):
        pieces = []
        total_chars = 0
        min_chars = int(os.environ.get(min_env, "120"))
        hard_chars = int(os.environ.get(max_env, "700"))

        for piece in stream:
            if not piece:
                continue
            pieces.append(piece)
            total_chars += len(piece)
            joined = "".join(pieces)
            if total_chars >= min_chars and joined.rstrip().endswith((".", "!", "?", "```")):
                break
            if total_chars >= hard_chars:
                break
        return "".join(pieces).strip()

    def _gemini_stream(self, contents, max_tokens=None):
        response = self._get_gemini_client().models.generate_content_stream(
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

    def generate_text_answer(self, custom_query=None):
        prompt = self._build_prompt(custom_query)
        return self._collect_fast(
            self._gemini_stream(prompt),
            "GEMINI_FAST_MIN_CHARS",
            "GEMINI_FAST_MAX_CHARS",
        )

    def generate_vision_answer(self, image_bytes, custom_query=None):
        from PIL import Image

        prompt = self._build_prompt(
            custom_query
            or "Solve or explain the coding, MCQ, diagram, or question shown in this image for practice."
        )
        image = Image.open(io.BytesIO(image_bytes))
        return self._collect_fast(
            self._gemini_stream(
                [prompt, image],
                max_tokens=int(os.environ.get("GEMINI_VISION_MAX_TOKENS", "768")),
            ),
            "GEMINI_FAST_MIN_CHARS",
            "GEMINI_FAST_MAX_CHARS",
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
