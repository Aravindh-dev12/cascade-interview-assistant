import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONTEXT_PATH = PROJECT_DIR / "data" / "candidate_context.local.md"


def _resolve_context_path():
    raw = os.environ.get("CANDIDATE_CONTEXT_FILE", "").strip()
    if not raw:
        return DEFAULT_CONTEXT_PATH
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path


def _read_private_context():
    path = _resolve_context_path()
    if not path.exists() or not path.is_file():
        return "", path
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        print(f"[context] Could not read candidate context {path}: {exc}")
        return "", path
    limit = max(2000, int(os.environ.get("CANDIDATE_CONTEXT_MAX_CHARS", "30000")))
    if len(text) > limit:
        text = text[:limit]
    return text, path


def install_candidate_context():
    """Augment CopilotAI with a git-ignored local candidate/resume/project context.

    The file is read on every request so the user can edit it without rebuilding or
    restarting. It is used for personal/project/behavioral answers; general technical
    questions still rely on the model's normal knowledge.
    """
    from engine.copilot_ai import CopilotAI

    if getattr(CopilotAI, "_candidate_context_installed", False):
        return

    original_knowledge_context = CopilotAI._knowledge_context
    original_build_prompt = CopilotAI._build_prompt
    announced = {"path": None, "loaded": False}

    def knowledge_context(ai):
        base = original_knowledge_context(ai)
        private_text, path = _read_private_context()
        if private_text:
            if announced["path"] != str(path) or not announced["loaded"]:
                print(f"[context] Candidate context loaded from {path}")
                announced["path"] = str(path)
                announced["loaded"] = True
            private_block = (
                "PRIVATE CANDIDATE CONTEXT (user-provided; use as the source of truth "
                "for first-person resume/project/experience answers; never invent missing facts):\n"
                + private_text
            )
            return (base + "\n\n" + private_block).strip() if base else private_block
        if announced["path"] != str(path):
            print(
                f"[context] No private candidate context found at {path}. "
                "General questions will still be answered normally."
            )
            announced["path"] = str(path)
            announced["loaded"] = False
        return base

    def build_prompt(ai, custom_query=None, image_task=None):
        prompt = original_build_prompt(ai, custom_query=custom_query, image_task=image_task)
        guidance = (
            "Answer policy:\n"
            "- For general technical/common interview questions, answer from model knowledge even if candidate context is empty.\n"
            "- For personal, behavioral, resume, project, metrics, employer, or experience questions, use only the supplied candidate context and transcript.\n"
            "- Never invent personal facts. If a required personal detail is absent, give the best concise answer framework and clearly avoid fabrication.\n"
            "- Keep answers candidate-ready: direct first, then only the strongest supporting details."
        )
        return guidance + "\n\n" + prompt

    CopilotAI._knowledge_context = knowledge_context
    CopilotAI._build_prompt = build_prompt
    CopilotAI._candidate_context_installed = True
