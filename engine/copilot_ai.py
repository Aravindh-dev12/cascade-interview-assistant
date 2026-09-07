import io
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
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


class CopilotAI:
    """Gemini text/vision engine with transcript and optional multi-frame practice context."""

    def __init__(self, model=None, api_key=None, provider=None):
        self.provider = "gemini"
        self.model = self._normalize_model(model)
        self.api_key = self._resolve_key(api_key)
        self._gemini_client = None
        self.interview_knowledge = _load_interview_knowledge()
        self.transcript_history = []
        self.max_transcript_history = max(4, int(os.environ.get("TRANSCRIPT_CONTEXT_LINES", "12")))
        self.image_history = []
        self.image_fingerprints = []
        self.max_image_history = max(1, int(os.environ.get("PRACTICE_IMAGE_CONTEXT_FRAMES", "3")))

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

    def set_config(self, model=None, api_key=None, provider=None):
        next_model = self._normalize_model(model)
        next_key = self._resolve_key(api_key)
        if next_key != self.api_key:
            self._gemini_client = None
        self.model = next_model
        self.api_key = next_key
        self.provider = "gemini"

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

    def generate_text_stream(self, custom_query=None):
        yield from self._gemini_stream(self._build_prompt(custom_query))

    def generate_vision_stream(self, image_bytes, custom_query=None, use_image_history=False):
        from PIL import Image

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
        contents = [prompt]
        for frame in frames:
            contents.append(Image.open(io.BytesIO(frame)).convert("RGB"))
        yield from self._gemini_stream(
            contents,
            max_tokens=int(os.environ.get("GEMINI_VISION_MAX_TOKENS", "1600")),
        )

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
            yield f"### Gemini Error\n\n`{exc}`"

    def generate_answer(self, image_bytes=None, custom_query=None, use_image_history=False):
        return "".join(
            self.generate_answer_stream(
                image_bytes=image_bytes,
                custom_query=custom_query,
                use_image_history=use_image_history,
            )
        ).strip()
