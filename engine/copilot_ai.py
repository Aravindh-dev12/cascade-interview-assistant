import base64
import io
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

load_dotenv()

DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5")
PROJECT_DIR = Path(__file__).resolve().parent.parent
KB_PATH = PROJECT_DIR / "data" / "interview_knowledge.json"

SYSTEM_PROMPT = (
    "You are a concise technical interview practice coach. Answer naturally in first person "
    "when the question is about the candidate's experience. Lead with the direct answer, then "
    "give 2-4 concrete supporting points. Never invent resume details, metrics, datasets, tools, "
    "or implementation facts. For coding questions, identify constraints, give the approach and "
    "compact correct code when the language is clear, then time and space complexity. For MCQs, "
    "state the best option first and briefly justify it. For math, solve carefully and verify the "
    "arithmetic. For architecture questions, reason from objective, observations, actions, state, "
    "uncertainty, planning, feedback, evaluation, and failure modes before naming frameworks."
)


def _practice_mode_enabled():
    return os.environ.get("PRACTICE_MODE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _load_interview_knowledge():
    try:
        with open(KB_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        print(f"[knowledge] Could not load interview KB: {exc}")
        return {}


class CopilotAI:
    """Practice answer engine supporting Gemini 2.5 Flash or OpenAI.

    NVIDIA speech-to-text is handled separately by STTWorker. The interview
    knowledge base is used only when PRACTICE_MODE is enabled.
    """

    def __init__(self, provider="gemini", model=None, api_key=None):
        self.provider = provider if provider in {"gemini", "openai"} else "gemini"
        self.model = self._normalize_model(self.provider, model)
        self.api_key = self._resolve_key(self.provider, api_key)
        self._gemini_client = None
        self._openai_client = None
        self.interview_knowledge = _load_interview_knowledge()
        self.transcript_history = []
        self.max_transcript_history = 12

    @staticmethod
    def _normalize_model(provider, model):
        model = str(model or "").strip()
        if provider == "openai":
            return model if model.startswith("gpt-") else DEFAULT_OPENAI_MODEL
        return model if model.startswith("gemini-2.5-") else DEFAULT_GEMINI_MODEL

    @staticmethod
    def _resolve_key(provider, api_key=None):
        supplied = (api_key or "").strip()
        if supplied:
            return supplied
        if provider == "openai":
            return os.environ.get("OPENAI_API_KEY", "").strip()
        return (
            os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        )

    def set_config(self, provider=None, model=None, api_key=None):
        next_provider = provider if provider in {"gemini", "openai"} else self.provider
        next_model = self._normalize_model(next_provider, model)
        next_key = self._resolve_key(next_provider, api_key)
        if next_provider != self.provider or next_key != self.api_key:
            self._gemini_client = None
            self._openai_client = None
        self.provider = next_provider
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

    def _knowledge_context(self):
        if not _practice_mode_enabled() or not self.interview_knowledge:
            return ""
        return json.dumps(self.interview_knowledge, ensure_ascii=False, indent=2)

    def _build_prompt(self, custom_query=None, image_task=None):
        transcript = self.get_formatted_transcript()
        task = custom_query or "Answer the latest substantive question in this mock interview transcript."
        knowledge = self._knowledge_context()

        parts = []
        if knowledge:
            parts.append(
                "Interview practice knowledge base. Treat this as candidate/role context. "
                "Use it when relevant but never invent facts beyond it:\n" + knowledge
            )
        parts.append("Conversation transcript:\n" + transcript)
        if image_task:
            parts.append("Image-analysis instructions:\n" + image_task)
        parts.append("Current task:\n" + task)
        return "\n\n".join(parts)

    def _get_gemini_client(self):
        if not self.api_key:
            self.api_key = self._resolve_key("gemini")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=self.api_key)
        return self._gemini_client

    def _get_openai_client(self):
        if not self.api_key:
            self.api_key = self._resolve_key("openai")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=self.api_key, timeout=20.0, max_retries=0)
        return self._openai_client

    def _collect_fast(self, stream, min_env, max_env):
        pieces = []
        total_chars = 0
        min_chars = int(os.environ.get(min_env, "140"))
        hard_chars = int(os.environ.get(max_env, "900"))
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
                or int(os.environ.get("GEMINI_TEXT_MAX_TOKENS", "512")),
            ),
        )
        for chunk in response:
            text = getattr(chunk, "text", None)
            if text:
                yield text

    def _openai_text(self, prompt):
        response = self._get_openai_client().responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=prompt,
            max_output_tokens=int(os.environ.get("OPENAI_TEXT_MAX_TOKENS", "512")),
        )
        return (response.output_text or "").strip()

    def _openai_vision(self, image_bytes, prompt):
        encoded = base64.b64encode(image_bytes).decode("ascii")
        response = self._get_openai_client().responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{encoded}"},
                    ],
                }
            ],
            max_output_tokens=int(os.environ.get("OPENAI_VISION_MAX_TOKENS", "1024")),
        )
        return (response.output_text or "").strip()

    def generate_text_answer(self, custom_query=None):
        prompt = self._build_prompt(custom_query)
        if self.provider == "openai":
            return self._openai_text(prompt)
        return self._collect_fast(
            self._gemini_stream(prompt),
            "GEMINI_FAST_MIN_CHARS",
            "GEMINI_FAST_MAX_CHARS",
        )

    def generate_vision_answer(self, image_bytes, custom_query=None):
        image_rules = self.interview_knowledge.get("practice_image_tasks", {}) if self.interview_knowledge else {}
        image_task = "\n".join(f"- {name}: {rule}" for name, rule in image_rules.items())
        prompt = self._build_prompt(
            custom_query or (
                "Identify exactly what the screenshot asks, then solve it. It may be a coding problem, "
                "HackerRank-style prompt, MCQ, mathematics problem, diagram, or system-design question. "
                "If text is unreadable or cropped, state what is missing instead of guessing."
            ),
            image_task=image_task,
        )

        if self.provider == "openai":
            return self._openai_vision(image_bytes, prompt)

        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self._collect_fast(
            self._gemini_stream(
                [prompt, image],
                max_tokens=int(os.environ.get("GEMINI_VISION_MAX_TOKENS", "1024")),
            ),
            "GEMINI_FAST_MIN_CHARS",
            "GEMINI_FAST_MAX_CHARS",
        )

    def generate_answer(self, image_bytes=None, custom_query=None):
        # In normal assessment mode this app remains transcription-only.
        if not _practice_mode_enabled():
            return (
                "### Live transcription active\n\n"
                "Practice coaching is disabled. Set `PRACTICE_MODE=1` only for mock interviews, "
                "practice, or sessions where AI assistance is explicitly permitted."
            )

        try:
            if image_bytes is not None:
                return self.generate_vision_answer(image_bytes, custom_query)
            return self.generate_text_answer(custom_query)
        except Exception as exc:
            provider_name = "OpenAI" if self.provider == "openai" else "Gemini"
            return f"### {provider_name} Error\n\n`{exc}`"
