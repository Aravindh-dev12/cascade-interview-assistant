import io
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = (
    "You are a fast technical interview practice coach. Give the useful answer first, "
    "then a short human-style explanation. Be decisive and concise. For MCQs, output "
    "the best option/answer on the first line, then one or two sentences explaining why. "
    "For coding questions, detect the requested programming language from the prompt or "
    "image and use it. Support Python, C, C++, Java, JavaScript, TypeScript, C#, Go, Rust, "
    "Kotlin, Swift, PHP, Ruby, SQL, Bash and other common languages. If no language is "
    "specified, prefer Python. Put the solution code before detailed explanation, followed "
    "by a compact approach and time/space complexity. For debugging questions, identify "
    "the bug and show the corrected code. For conceptual questions, answer naturally as "
    "a strong candidate would speak: clear, direct, and not robotic. Do not add filler or "
    "repeat the question."
)


def _practice_mode_enabled():
    return os.environ.get("PRACTICE_MODE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class CopilotAI:
    """Gemini-only answer engine for text, chat, and screen/vision prompts."""

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
        task = custom_query or (
            "Answer the latest substantive interviewer question for a mock/practice "
            "interview. Return the direct candidate-ready answer first."
        )
        return f"Transcript:\n{transcript}\n\nTask:\n{task}"

    def _get_gemini_client(self):
        if not self.api_key:
            self.api_key = self._resolve_key()
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured in the project .env")
        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=self.api_key)
        return self._gemini_client

    def _collect_fast(self, stream, min_env, max_env):
        pieces = []
        total_chars = 0
        min_chars = int(os.environ.get(min_env, "80"))
        hard_chars = int(os.environ.get(max_env, "650"))

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
        thinking_budget = int(os.environ.get("GEMINI_THINKING_BUDGET", "0"))
        response = self._get_gemini_client().models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=max_tokens
                or int(os.environ.get("GEMINI_TEXT_MAX_TOKENS", "320")),
                thinking_config=types.ThinkingConfig(
                    thinking_budget=thinking_budget,
                ),
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
            or (
                "Analyze the screenshot immediately. Determine whether it is an MCQ, "
                "coding problem, debugging task, output question, diagram, or conceptual "
                "question. For an MCQ, give the correct option first. For coding, detect "
                "the requested language and return working code first; if no language is "
                "specified use Python. Then give only the essential explanation and "
                "complexity. Read all visible constraints and examples carefully."
            )
        )
        image = Image.open(io.BytesIO(image_bytes))
        return self._collect_fast(
            self._gemini_stream(
                [prompt, image],
                max_tokens=int(os.environ.get("GEMINI_VISION_MAX_TOKENS", "640")),
            ),
            "GEMINI_FAST_MIN_CHARS",
            "GEMINI_FAST_MAX_CHARS",
        )

    def generate_answer(self, image_bytes=None, custom_query=None):
        # Only hands-free transcript coaching is practice-mode gated. Explicit user
        # actions (typed chat and screen capture) remain available independently.
        if image_bytes is None and custom_query is None and not _practice_mode_enabled():
            return (
                "### Automatic coaching is off\n\n"
                "Set `PRACTICE_MODE=1` in `.env` for mock interviews or sessions "
                "where AI assistance is explicitly permitted. Chat and screen capture "
                "still work as explicit actions."
            )

        try:
            if image_bytes is not None:
                return self.generate_vision_answer(image_bytes, custom_query)
            return self.generate_text_answer(custom_query)
        except Exception as exc:
            return f"### Gemini Error\n\n`{exc}`"
