import os
from pathlib import Path

from utils.private_context_store import (
    context_store_status,
    import_file,
    search_context,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONTEXT_PATH = PROJECT_DIR / "data" / "candidate_context.local.md"
DEFAULT_PROJECTS_PATH = PROJECT_DIR / "data" / "candidate_projects.local.md"
_SYNCED_MTIMES = {}


def _resolve_context_path():
    raw = os.environ.get("CANDIDATE_CONTEXT_FILE", "").strip()
    if not raw:
        return DEFAULT_CONTEXT_PATH
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path


def _resolve_projects_path():
    raw = os.environ.get("CANDIDATE_PROJECTS_FILE", "").strip()
    if not raw:
        return DEFAULT_PROJECTS_PATH
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path


def _candidate_source_paths():
    paths = [_resolve_context_path(), _resolve_projects_path()]
    for fallback in (
        PROJECT_DIR / "data" / "candidate_context.local.txt",
        PROJECT_DIR / "data" / "candidate_context.local.json",
    ):
        if fallback not in paths:
            paths.append(fallback)
    return paths


def _sync_private_sources():
    imported = 0
    for path in _candidate_source_paths():
        if not path.exists() or not path.is_file():
            continue
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            continue
        key = str(path.resolve())
        if _SYNCED_MTIMES.get(key) == mtime:
            continue
        try:
            imported += import_file(path, source=key)
            _SYNCED_MTIMES[key] = mtime
            print(f"[context-db] Imported private context from {path}")
        except Exception as exc:
            print(f"[context-db] Could not import {path}: {exc}")
    return imported


def _legacy_context_text():
    """Fallback only if the encrypted store is unavailable or empty."""
    path = _resolve_context_path()
    if not path.exists() or not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        print(f"[context] Could not read candidate context {path}: {exc}")
        return ""
    limit = max(2000, int(os.environ.get("CANDIDATE_CONTEXT_MAX_CHARS", "30000")))
    return text[:limit]


def install_candidate_context():
    """Add private, local-only candidate/project context to Gemini prompts.

    Local markdown/text files are imported into a SQLite store outside the repository.
    On Windows, each stored context chunk is protected with the current Windows user's
    DPAPI credentials. Only relevant chunks are retrieved for each question so large
    project histories do not bloat every prompt.
    """
    from engine.copilot_ai import CopilotAI

    if getattr(CopilotAI, "_candidate_context_installed", False):
        return

    original_build_prompt = CopilotAI._build_prompt
    announced = {"store": False}

    _sync_private_sources()

    def build_prompt(ai, custom_query=None, image_task=None):
        _sync_private_sources()

        prompt = original_build_prompt(
            ai,
            custom_query=custom_query,
            image_task=image_task,
        )
        recent_transcript = "\n".join(
            f"{item.get('speaker', '')}: {item.get('text', '')}"
            for item in ai.transcript_history[-6:]
        )
        retrieval_query = "\n".join(
            part
            for part in (
                str(custom_query or ""),
                str(image_task or ""),
                recent_transcript,
            )
            if part.strip()
        )

        max_chunks = max(1, min(20, int(os.environ.get("PRIVATE_CONTEXT_MAX_CHUNKS", "8"))))
        max_chars = max(4000, min(50000, int(os.environ.get("PRIVATE_CONTEXT_MAX_CHARS", "18000"))))
        private_text = search_context(
            retrieval_query,
            limit=max_chunks,
            max_chars=max_chars,
        )
        if not private_text:
            private_text = _legacy_context_text()

        if not announced["store"]:
            status = context_store_status()
            print(
                f"[context-db] Private store: {status['path']} · chunks={status['chunks']} · "
                f"protection={status['protection']}"
            )
            announced["store"] = True

        guidance = (
            "Answer policy:\n"
            "- For general technical/common practice questions, answer from model knowledge.\n"
            "- For personal, behavioral, resume, previous-company, project, metrics, employer, or experience questions, "
            "  use only the supplied private candidate context and transcript for personal facts.\n"
            "- Never invent personal facts, responsibilities, metrics, employers, production claims, or project details.\n"
            "- If the context does not support a requested personal detail, say what is known and avoid fabrication.\n"
            "- Keep answers candidate-ready: direct first, then only the strongest supporting details."
        )

        if private_text:
            private_block = (
                "PRIVATE CANDIDATE/PROJECT CONTEXT (local user-provided source of truth):\n"
                + private_text
            )
            return guidance + "\n\n" + private_block + "\n\n" + prompt
        return guidance + "\n\n" + prompt

    CopilotAI._build_prompt = build_prompt
    CopilotAI._candidate_context_installed = True
