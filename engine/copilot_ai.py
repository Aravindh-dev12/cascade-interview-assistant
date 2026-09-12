import io
import json
import os
import time
from pathlib import Path

from google import genai
from google.genai import types

import config
from engine.nvidia_omni_vision import NvidiaOmniVisionClient

PROJECT_DIR = Path(__file__).resolve().parent.parent
KB_PATH = PROJECT_DIR / "data" / "interview_knowledge.json"

SYSTEM_PROMPT = (
    "You are a low-latency technical interview practice coach. Give the candidate-ready answer immediately. "
    "Be concise, natural, and decisive. For conceptual questions, start with a 20-40 second spoken answer, "
    "then add only essential supporting points. For MCQs, put the best option on the first line. For coding "
    "questions, detect the requested language, give the approach briefly, then complete correct code, then "
    "time/space complexity. For debugging, identify the bug and provide corrected code. Use supplied screen "
    "or camera evidence carefully. Never invent personal experience, resume facts, metrics, project details, "
    "or unreadable visual text. Do not repeat the question and do not add filler."
)


def _practice_mode_enabled():
    return os.environ.get("PRACTICE_MODE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _load_interview_knowledge():
    try:
        with open(KB_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        print(f"[knowledge] Could not load interview KB: {exc}")
        return {}


def _gemini_key():
    return (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )


def _mime_type(image_bytes):
    if bytes(image_bytes or b"")[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return "image/jpeg"


class CopilotAI:
    """Gemini 2.5 Flash final-answer engine with NVIDIA visual extraction.

    Text path:
        NVIDIA Parakeet transcript -> Gemini 2.5 Flash -> overlay.

    Visual path:
        Manual screen/camera -> NVIDIA Nemotron Omni extraction -> Gemini 2.5 Flash.
        If NVIDIA vision is unavailable or returns unusable text, the original image is
        sent directly to Gemini 2.5 Flash. There is no Ollama/local-model path.
    """

    def __init__(self, provider=None, model=None, api_key=None):
        settings = config.load_settings()
        self.provider_preference = "gemini"
        self.model = config.DEFAULT_GEMINI_MODEL
        self.api_key = (api_key or _gemini_key()).strip()
        self._gemini_client = None
        self._client_key = None
        self._active_provider = "gemini"

        self.interview_knowledge = _load_interview_knowledge()
        self.transcript_history = []
        self.max_transcript_history = max(
            4, int(os.environ.get("TRANSCRIPT_CONTEXT_LINES", "12"))
        )
        self.image_history = []
        self.image_fingerprints = []
        self.max_image_history = max(
            1, int(os.environ.get("PRACTICE_IMAGE_CONTEXT_FRAMES", "1"))
        )
        self.omni_vision = NvidiaOmniVisionClient(
            api_key=os.environ.get("NVIDIA_API_KEY", "")
        )

        # Ignore stale saved provider/model values. This build is intentionally fixed
        # to Gemini 2.5 Flash for final answers.
        self.set_config(
            provider="gemini",
            model=settings.get("model", config.DEFAULT_GEMINI_MODEL),
            api_key=self.api_key,
        )
        print(
            f"[pipeline] Final answer engine: Google {self.model}; "
            f"vision extractor: {self.omni_vision.model}; local models disabled"
        )

    def set_config(self, provider=None, model=None, api_key=None):
        self.provider_preference = "gemini"
        self.model = config.DEFAULT_GEMINI_MODEL
        next_key = (api_key or _gemini_key()).strip()
        if next_key != self.api_key:
            self.api_key = next_key
            self._gemini_client = None
            self._client_key = None
        self.omni_vision.reconfigure()

    def runtime_label(self):
        if not _gemini_key():
            return f"GEMINI KEY MISSING · {self.model} · NVIDIA Parakeet/vision"
        return f"GEMINI · {self.model} · NVIDIA Parakeet + vision"

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
        return "\n".join(
            f"{item['speaker']}: {item['text']}" for item in self.transcript_history
        )

    def _knowledge_context(self):
        if not _practice_mode_enabled() or not self.interview_knowledge:
            return ""
        return json.dumps(
            self.interview_knowledge,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _build_prompt(self, custom_query=None, image_task=None):
        task = custom_query or (
            "Answer the latest substantive interviewer question in the practice transcript."
        )
        parts = []
        knowledge = self._knowledge_context()
        if knowledge:
            parts.append(
                "Candidate/role practice context follows. Use only when relevant; never invent beyond it:\n"
                + knowledge
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
        return sum(abs(a - b) for a, b in zip(left, right)) / (
            255.0 * len(left)
        )

    def _remember_image(self, image_bytes):
        try:
            fp = self._image_fingerprint(image_bytes)
        except Exception:
            fp = b""
        duplicate_threshold = float(
            os.environ.get("PRACTICE_IMAGE_DUPLICATE_THRESHOLD", "0.012")
        )
        if self.image_fingerprints and fp:
            if (
                self._fingerprint_distance(self.image_fingerprints[-1], fp)
                <= duplicate_threshold
            ):
                self.image_history[-1] = image_bytes
                self.image_fingerprints[-1] = fp
                return
        self.image_history.append(image_bytes)
        self.image_fingerprints.append(fp)
        if len(self.image_history) > self.max_image_history:
            del self.image_history[:-self.max_image_history]
            del self.image_fingerprints[:-self.max_image_history]

    def _get_gemini_client(self):
        key = _gemini_key()
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing from the project .env file. "
                "Create a Google Gemini API key and set GEMINI_API_KEY locally; do not paste it into chat or commit it."
            )
        if self._gemini_client is None or self._client_key != key:
            self._gemini_client = genai.Client(api_key=key)
            self._client_key = key
        return self._gemini_client

    def _generation_config(self, max_tokens):
        kwargs = {
            "system_instruction": SYSTEM_PROMPT,
            "max_output_tokens": int(max_tokens),
            "temperature": float(os.environ.get("GEMINI_TEMPERATURE", "0.15")),
        }
        # Gemini 2.5 Flash supports thinking. A zero budget is intentional here because
        # the product requirement is immediate practice answers rather than long reasoning.
        try:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass
        return types.GenerateContentConfig(**kwargs)

    def _gemini_stream(self, prompt, max_tokens, image_bytes_list=None):
        client = self._get_gemini_client()
        contents = []
        for image_bytes in image_bytes_list or []:
            if not image_bytes:
                continue
            contents.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=_mime_type(image_bytes),
                )
            )
        contents.append(str(prompt))

        started = time.monotonic()
        first = True
        try:
            response = client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=self._generation_config(max_tokens),
            )
            for chunk in response:
                text = getattr(chunk, "text", None)
                if not text:
                    continue
                if first:
                    first = False
                    mode = "vision" if image_bytes_list else "text"
                    print(
                        f"[gemini] First {mode} answer token in "
                        f"{time.monotonic() - started:.2f}s · model={self.model}"
                    )
                yield text
            if first:
                raise RuntimeError("Gemini returned no answer text")
        except Exception as exc:
            raise RuntimeError(f"Gemini {self.model} request failed: {exc}") from exc

    @staticmethod
    def _validate_visual_text(text):
        cleaned = str(text or "").strip()
        try:
            configured_min = int(
                os.environ.get("NVIDIA_VISION_MIN_USEFUL_CHARS", "32")
            )
        except Exception:
            configured_min = 32
        minimum = max(12, min(200, configured_min))
        label_only = {
            "mcq",
            "coding",
            "debug",
            "debugging",
            "general",
            "image",
            "screen",
            "unknown",
            "none",
        }
        if len(cleaned) < minimum or cleaned.lower().strip(" .:-") in label_only:
            raise RuntimeError(
                f"NVIDIA Omni extraction was too short to answer safely "
                f"({len(cleaned)} chars; need at least {minimum})"
            )
        return cleaned

    def _route_stream(self, prompt, max_tokens, image_bytes_list=None):
        # Compatibility hook for any legacy caller. Final generation is always Gemini.
        yield from self._gemini_stream(
            prompt,
            max_tokens=max_tokens,
            image_bytes_list=image_bytes_list,
        )

    def generate_text_stream(self, custom_query=None):
        self.set_config(provider="gemini")
        prompt = self._build_prompt(custom_query)
        max_tokens = int(os.environ.get("TEXT_MAX_TOKENS", "700"))
        print(
            f"[pipeline] Parakeet transcript/chat -> Gemini 2.5 Flash · "
            f"prompt_chars={len(prompt)}"
        )
        yield from self._gemini_stream(prompt, max_tokens=max_tokens)

    def generate_vision_stream(
        self,
        image_bytes,
        custom_query=None,
        use_image_history=False,
    ):
        self.set_config(provider="gemini")
        task = custom_query or (
            "Solve or explain the current visible practice question. Give the useful answer first."
        )
        max_tokens = int(os.environ.get("VISION_MAX_TOKENS", "1000"))

        if use_image_history:
            self._remember_image(image_bytes)

        print(
            f"[pipeline] Manual image -> NVIDIA Omni extraction · "
            f"bytes={len(image_bytes or b'')}"
        )
        try:
            visual_text = self.omni_vision.extract(image_bytes, task_hint=task)
            visual_text = self._validate_visual_text(visual_text)
            prompt = (
                "Solve the practice question from the NVIDIA-extracted screen text below. "
                "Answer immediately; do not repeat the problem statement. For MCQ put the correct option "
                "first. For coding give a short approach, complete correct code in the requested language, "
                "then time and space complexity. For debugging state the defect and corrected code. "
                "Do not invent missing visual details.\n\n"
                f"USER TASK:\n{task}\n\n"
                f"NVIDIA VISUAL EXTRACTION:\n{visual_text}"
            )
            print(
                f"[pipeline] NVIDIA visual text -> Gemini 2.5 Flash · "
                f"screen_chars={len(visual_text)} · prompt_chars={len(prompt)}"
            )
            yield from self._gemini_stream(prompt, max_tokens=max_tokens)
            return
        except Exception as exc:
            print(f"[vision] NVIDIA extraction unusable/unavailable: {exc}")
            print(
                "[vision] Falling back to direct Gemini 2.5 Flash image analysis for this capture."
            )

        fallback_prompt = (
            "Read the supplied practice screenshot carefully and answer the visible question immediately. "
            "For MCQ put the correct option first. For coding give a short approach, complete correct code "
            "in the requested language, then time and space complexity. For debugging identify the defect "
            "and corrected code. Do not invent unreadable text.\n\n"
            f"USER TASK:\n{task}"
        )
        yield from self._gemini_stream(
            fallback_prompt,
            max_tokens=max_tokens,
            image_bytes_list=[image_bytes],
        )

    def generate_answer_stream(
        self,
        image_bytes=None,
        custom_query=None,
        use_image_history=False,
    ):
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

    def generate_answer(
        self,
        image_bytes=None,
        custom_query=None,
        use_image_history=False,
    ):
        return "".join(
            self.generate_answer_stream(
                image_bytes=image_bytes,
                custom_query=custom_query,
                use_image_history=use_image_history,
            )
        ).strip()
