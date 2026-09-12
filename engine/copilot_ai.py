import io
import json
import os
import re
import time
from pathlib import Path

from google import genai
from google.genai import types

import config
from engine.nvidia_omni_vision import NvidiaOmniVisionClient

PROJECT_DIR = Path(__file__).resolve().parent.parent
KB_PATH = PROJECT_DIR / "data" / "interview_knowledge.json"

SYSTEM_PROMPT = (
    "You are a low-latency technical interview practice coach. Give the useful answer immediately. "
    "For MCQs, put the correct option letter/number and answer on the first line, then one short reason. "
    "For coding problems, obey the programming language currently selected or explicitly requested on screen. "
    "If no language is visible or requested, default to Python 3. Give a short approach, complete runnable or "
    "platform-compatible code, then time and space complexity and a brief correctness/edge-case explanation. "
    "Do not merely fit sample tests: reason from constraints and handle boundary cases, duplicates, empty/single "
    "inputs where valid, negative/large values, indexing, parsing, overflow where relevant, and the exact required "
    "function signature or stdin/stdout contract. For debugging, identify the defect and provide corrected code. "
    "Use supplied screen/camera evidence carefully. Never invent unreadable visual text or missing personal facts. "
    "Do not repeat the entire question and do not add filler."
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


def _bounded_int_env(name, default, minimum, maximum):
    try:
        value = int(os.environ.get(name, str(default)))
    except Exception:
        value = int(default)
    return max(int(minimum), min(int(maximum), value))


class CopilotAI:
    """Gemini 2.5 Flash final-answer engine with NVIDIA perception.

    Text path:
        NVIDIA Parakeet transcript -> Gemini 2.5 Flash -> overlay.

    Visual path:
        Manual screen/camera -> NVIDIA Nemotron Omni extraction -> Gemini 2.5 Flash.
        Repeated manual captures can be accumulated as chronological scroll snapshots.
        If NVIDIA vision is unavailable or unusable, recent captured images are sent
        directly to Gemini 2.5 Flash. There is no Ollama/local-model path.
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

        # Raw image history is only used when NVIDIA extraction is unavailable.
        self.image_history = []
        self.image_fingerprints = []
        self.max_image_history = _bounded_int_env(
            "PRACTICE_IMAGE_CONTEXT_FRAMES", 6, 1, 8
        )

        # NVIDIA text extractions are much cheaper to accumulate than full images and
        # let a long HackerRank-style problem be reconstructed over several scrolls.
        self.visual_text_history = []
        self.max_visual_text_history = _bounded_int_env(
            "PRACTICE_VISUAL_TEXT_HISTORY", 6, 1, 10
        )
        self.last_problem_title = ""

        self.omni_vision = NvidiaOmniVisionClient(
            api_key=os.environ.get("NVIDIA_API_KEY", "")
        )

        self.set_config(
            provider="gemini",
            model=settings.get("model", config.DEFAULT_GEMINI_MODEL),
            api_key=self.api_key,
        )
        print(
            f"[pipeline] Final answer engine: Google {self.model}; "
            f"vision extractor: {self.omni_vision.model}; scroll history={self.max_visual_text_history}; "
            "local models disabled"
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
        return f"GEMINI · {self.model} · NVIDIA Parakeet + scroll-aware vision"

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
        self.visual_text_history.clear()
        self.last_problem_title = ""

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

    @staticmethod
    def _visual_field(text, name):
        match = re.search(
            rf"(?im)^\s*{re.escape(name)}\s*:\s*(.*?)\s*$",
            str(text or ""),
        )
        return match.group(1).strip() if match else ""

    @staticmethod
    def _normalized_title(value):
        value = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
        if value in {"", "unknown", "none", "n a", "na"}:
            return ""
        return value

    def _reset_visual_problem(self, image_bytes=None):
        self.visual_text_history.clear()
        self.image_history.clear()
        self.image_fingerprints.clear()
        self.last_problem_title = ""
        if image_bytes:
            self._remember_image(image_bytes)

    def _remember_visual_text(self, visual_text, image_bytes=None):
        """Add one scroll snapshot and reset when a clearly different titled problem appears."""
        current_title = self._normalized_title(
            self._visual_field(visual_text, "PROBLEM_TITLE")
        )
        previous_title = self._normalized_title(self.last_problem_title)
        if current_title and previous_title and current_title != previous_title:
            # Only reset on a meaningful title change. Generic/unknown titles are ignored.
            current_words = set(current_title.split())
            previous_words = set(previous_title.split())
            overlap = len(current_words & previous_words) / max(
                1, min(len(current_words), len(previous_words))
            )
            if overlap < 0.5:
                print(
                    f"[vision] New problem title detected; clearing old scroll context · "
                    f"old={self.last_problem_title!r} · new={self._visual_field(visual_text, 'PROBLEM_TITLE')!r}"
                )
                self._reset_visual_problem(image_bytes=image_bytes)

        if current_title:
            self.last_problem_title = self._visual_field(visual_text, "PROBLEM_TITLE")

        cleaned = str(visual_text or "").strip()
        if not cleaned:
            return
        if self.visual_text_history and cleaned == self.visual_text_history[-1]:
            self.visual_text_history[-1] = cleaned
        else:
            self.visual_text_history.append(cleaned)
        if len(self.visual_text_history) > self.max_visual_text_history:
            del self.visual_text_history[:-self.max_visual_text_history]

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

    def _generation_config(self, max_tokens, thinking_budget=0):
        kwargs = {
            "system_instruction": SYSTEM_PROMPT,
            "max_output_tokens": int(max_tokens),
            "temperature": float(os.environ.get("GEMINI_TEMPERATURE", "0.15")),
        }
        try:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=max(0, int(thinking_budget))
            )
        except Exception:
            pass
        return types.GenerateContentConfig(**kwargs)

    def _gemini_stream(
        self,
        prompt,
        max_tokens,
        image_bytes_list=None,
        thinking_budget=0,
    ):
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
                config=self._generation_config(
                    max_tokens,
                    thinking_budget=thinking_budget,
                ),
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
                        f"{time.monotonic() - started:.2f}s · model={self.model} · "
                        f"thinking={thinking_budget}"
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

    @staticmethod
    def _visual_thinking_budget(visual_text):
        problem_type = CopilotAI._visual_field(visual_text, "TYPE").lower()
        if any(token in problem_type for token in ("coding", "debug", "sql")):
            return _bounded_int_env(
                "GEMINI_CODING_THINKING_BUDGET", 1024, 0, 4096
            )
        return _bounded_int_env("GEMINI_VISUAL_THINKING_BUDGET", 256, 0, 2048)

    def _scroll_solver_prompt(self, task, snapshots):
        blocks = []
        for index, text in enumerate(snapshots, start=1):
            blocks.append(f"--- CAPTURE {index} ---\n{text}")
        combined = "\n\n".join(blocks)
        return (
            "Solve the current practice question using the chronological screen captures below. "
            "They may be overlapping scroll snapshots of one long problem. Merge overlapping statement text, "
            "constraints, examples, starter code, and options instead of treating each capture as a separate question. "
            "The NEWEST capture is authoritative for UI state, especially ACTIVE_LANGUAGE, language dropdowns, "
            "selected tabs, visible function signatures, and changed answer choices. If the newest capture clearly "
            "shows a different problem from older captures, ignore the unrelated older captures.\n\n"
            "Answer rules:\n"
            "- MCQ: first line must be `Answer: <option> — <answer text>`, then a concise reason. A visibly selected "
            "  option is only UI state; independently determine whether it is correct.\n"
            "- Coding: use ACTIVE_LANGUAGE from the newest capture when present. If no language is visible anywhere, "
            "  default to Python 3. Give: Approach, complete Code, Complexity, then Edge cases/correctness. Preserve the "
            "  required function signature or input/output format. Derive the algorithm from constraints and design for "
            "  hidden/boundary tests, not only the samples.\n"
            "- Debugging/SQL/output: give the corrected/direct answer first and a short explanation.\n"
            "- If the captures are genuinely missing a required part of the problem and a reliable answer cannot be "
            "  produced yet, say `NEED_MORE_SCREEN` and name exactly what needs to be captured next instead of inventing it.\n\n"
            f"USER TASK:\n{task}\n\n"
            f"SCROLL CAPTURES ({len(snapshots)} total):\n{combined}"
        )

    def _route_stream(self, prompt, max_tokens, image_bytes_list=None):
        yield from self._gemini_stream(
            prompt,
            max_tokens=max_tokens,
            image_bytes_list=image_bytes_list,
        )

    def generate_text_stream(self, custom_query=None):
        self.set_config(provider="gemini")
        prompt = self._build_prompt(custom_query)
        max_tokens = int(os.environ.get("TEXT_MAX_TOKENS", "900"))
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
        max_tokens = int(os.environ.get("VISION_MAX_TOKENS", "1800"))

        if use_image_history:
            self._remember_image(image_bytes)

        print(
            f"[pipeline] Manual image -> NVIDIA Omni extraction · "
            f"bytes={len(image_bytes or b'')} · scroll={bool(use_image_history)}"
        )
        try:
            visual_text = self.omni_vision.extract(image_bytes, task_hint=task)
            visual_text = self._validate_visual_text(visual_text)

            if use_image_history:
                self._remember_visual_text(visual_text, image_bytes=image_bytes)
                snapshots = list(self.visual_text_history)
            else:
                snapshots = [visual_text]

            latest = snapshots[-1]
            thinking_budget = self._visual_thinking_budget(latest)
            prompt = self._scroll_solver_prompt(task, snapshots)
            print(
                f"[pipeline] NVIDIA scroll context -> Gemini 2.5 Flash · "
                f"captures={len(snapshots)} · latest_chars={len(latest)} · "
                f"prompt_chars={len(prompt)}"
            )
            yield from self._gemini_stream(
                prompt,
                max_tokens=max_tokens,
                thinking_budget=thinking_budget,
            )
            return
        except Exception as exc:
            print(f"[vision] NVIDIA extraction unusable/unavailable: {exc}")
            print(
                "[vision] Falling back to direct Gemini 2.5 Flash image analysis for this capture/session."
            )

        frames = (
            list(self.image_history[-self.max_image_history :])
            if use_image_history and self.image_history
            else [image_bytes]
        )
        fallback_prompt = (
            "These images are chronological practice-screen captures and may be overlapping scroll snapshots of "
            "one long question. Combine all readable parts. The newest image is authoritative for the selected "
            "programming language and current UI state. For MCQ return the correct option first and explain briefly. "
            "For coding, use the visible selected/requested language (Python 3 only if none is visible), give a short "
            "approach, complete platform-compatible code, complexity, and edge-case/correctness notes. Design from the "
            "constraints for hidden/boundary tests rather than only samples. If essential text is still missing, reply "
            "NEED_MORE_SCREEN and state what to capture next. Do not invent unreadable text.\n\n"
            f"USER TASK:\n{task}"
        )
        print(
            f"[pipeline] Direct Gemini scroll vision fallback · frames={len(frames)}"
        )
        yield from self._gemini_stream(
            fallback_prompt,
            max_tokens=max_tokens,
            image_bytes_list=frames,
            thinking_budget=_bounded_int_env(
                "GEMINI_CODING_THINKING_BUDGET", 1024, 0, 4096
            ),
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
