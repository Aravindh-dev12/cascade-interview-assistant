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
    "You are a concise technical interview practice coach. Answer naturally in first person "
    "when the question is about the candidate's experience. Lead with the direct answer, then "
    "give 2-4 concrete supporting points. Never invent resume details, metrics, datasets, tools, "
    "or implementation facts. For coding questions, first reconstruct the complete problem and "
    "constraints, then give the approach and compact correct code when the language is clear, "
    "followed by time and space complexity. For MCQs, state the best option first and briefly "
    "justify it. For math, solve carefully and verify arithmetic. For architecture questions, "
    "reason from objective, observations, actions, state, uncertainty, planning, feedback, "
    "evaluation, and failure modes before naming frameworks."
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
    """Gemini-only practice answer and vision engine.

    NVIDIA speech-to-text is handled separately by STTWorker. Interview context
    and multi-frame screenshot accumulation are used only in practice/permitted mode.
    """

    def __init__(self, provider="gemini", model=None, api_key=None):
        self.provider = "gemini"
        self.model = self._normalize_model(model)
        self.api_key = self._resolve_key(api_key)
        self._gemini_client = None
        self.interview_knowledge = _load_interview_knowledge()
        self.transcript_history = []
        self.max_transcript_history = 12
        self.image_history = []
        self.image_fingerprints = []
        self.max_image_history = max(
            1, int(os.environ.get("PRACTICE_IMAGE_CONTEXT_FRAMES", "6"))
        )

    @staticmethod
    def _normalize_model(model):
        model = str(model or "").strip()
        return model if model.startswith("gemini-2.5-") else DEFAULT_GEMINI_MODEL

    @staticmethod
    def _resolve_key(api_key=None):
        supplied = (api_key or "").strip()
        if supplied:
            return supplied
        return (
            os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        )

    def set_config(self, provider=None, model=None, api_key=None):
        next_model = self._normalize_model(model)
        next_key = self._resolve_key(api_key)
        if next_key != self.api_key:
            self._gemini_client = None
        self.provider = "gemini"
        self.model = next_model
        self.api_key = next_key

    def add_transcript_line(self, speaker, text):
        self.transcript_history.append({"speaker": speaker, "text": text})
        if len(self.transcript_history) > self.max_transcript_history:
            self.transcript_history.pop(0)

    def clear_history(self):
        self.transcript_history.clear()
        self.image_history.clear()
        self.image_fingerprints.clear()

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
                "Use it for both common interview questions and role-specific questions when relevant, "
                "but never invent facts beyond it:\n" + knowledge
            )
        parts.append("Conversation transcript:\n" + transcript)
        if image_task:
            parts.append("Image-analysis instructions:\n" + image_task)
        parts.append("Current task:\n" + task)
        return "\n\n".join(parts)

    @staticmethod
    def _image_fingerprint(image_bytes):
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("L").resize((32, 32))
        return bytes(image.tobytes())

    @staticmethod
    def _fingerprint_distance(left, right):
        if not left or not right or len(left) != len(right):
            return 1.0
        total = sum(abs(a - b) for a, b in zip(left, right))
        return total / (255.0 * len(left))

    def _remember_image(self, image_bytes):
        try:
            fingerprint = self._image_fingerprint(image_bytes)
        except Exception:
            fingerprint = b""

        duplicate_threshold = float(
            os.environ.get("PRACTICE_IMAGE_DUPLICATE_THRESHOLD", "0.012")
        )
        if self.image_fingerprints and fingerprint:
            distance = self._fingerprint_distance(
                self.image_fingerprints[-1], fingerprint
            )
            if distance <= duplicate_threshold:
                self.image_history[-1] = image_bytes
                self.image_fingerprints[-1] = fingerprint
                return len(self.image_history)

        self.image_history.append(image_bytes)
        self.image_fingerprints.append(fingerprint)
        if len(self.image_history) > self.max_image_history:
            self.image_history.pop(0)
            self.image_fingerprints.pop(0)
        return len(self.image_history)

    def _get_gemini_client(self):
        if not self.api_key:
            self.api_key = self._resolve_key()
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=self.api_key)
        return self._gemini_client

    def _collect_fast(self, stream):
        pieces = []
        total_chars = 0
        min_chars = int(os.environ.get("GEMINI_FAST_MIN_CHARS", "140"))
        hard_chars = int(os.environ.get("GEMINI_FAST_MAX_CHARS", "1200"))
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

    def generate_text_answer(self, custom_query=None):
        return self._collect_fast(self._gemini_stream(self._build_prompt(custom_query)))

    def generate_vision_answer(self, image_bytes, custom_query=None):
        frame_count = self._remember_image(image_bytes)
        image_rules = self.interview_knowledge.get("practice_image_tasks", {}) if self.interview_knowledge else {}
        image_task = "\n".join(f"- {name}: {rule}" for name, rule in image_rules.items())
        image_task += (
            f"\n- scrolling-context: You are receiving {frame_count} chronological screenshot frame(s) "
            "from the same practice problem/session, oldest first and newest last. Reconstruct one complete "
            "problem by combining text across all frames, deduplicating overlapping sections caused by scrolling. "
            "Preserve the exact visible constraints, examples, function signature, starter code, and requested "
            "output format. Do not treat the newest frame as a separate problem unless the content clearly changed. "
            "If essential information is still missing, say exactly which section should be captured next instead "
            "of guessing."
        )
        prompt = self._build_prompt(
            custom_query or (
                "Identify exactly what the combined screenshots ask, reconstruct the full problem across scrolls, "
                "then solve it. It may be a HackerRank-style coding prompt, MCQ, mathematics problem, diagram, "
                "or system-design question."
            ),
            image_task=image_task,
        )

        from PIL import Image

        contents = [prompt]
        for frame in self.image_history:
            contents.append(Image.open(io.BytesIO(frame)).convert("RGB"))

        print(f"[vision] Using {len(self.image_history)} practice screenshot frame(s) as context.")
        return self._collect_fast(
            self._gemini_stream(
                contents,
                max_tokens=int(os.environ.get("GEMINI_VISION_MAX_TOKENS", "1400")),
            )
        )

    def generate_answer(self, image_bytes=None, custom_query=None):
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
            return f"### Gemini Error\n\n`{exc}`"
