import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv


SUPPORTED_ENV_FILES = (".env", ".env.local", ".env.txt", "env")


def load_project_env(project_dir: Path):
    """Load project-local runtime configuration without printing secret values."""
    project_dir = Path(project_dir).resolve()
    selected = None

    for name in SUPPORTED_ENV_FILES:
        candidate = project_dir / name
        if candidate.is_file():
            selected = candidate
            break

    if selected is not None:
        load_dotenv(selected, override=True)
        parsed = dotenv_values(selected)
    else:
        parsed = {}

    detected_names = []
    for name in (
        "NVIDIA_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "AI_PROVIDER",
        "GEMINI_MODEL",
        "PRACTICE_MODE",
    ):
        if os.environ.get(name, "").strip() or (parsed and parsed.get(name)):
            detected_names.append(name)

    gemini_loaded = bool(
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )

    return {
        "selected_path": selected,
        "exists": selected is not None,
        "detected_names": detected_names,
        "nvidia_loaded": bool(os.environ.get("NVIDIA_API_KEY", "").strip()),
        "gemini_loaded": gemini_loaded,
        "practice_mode": os.environ.get("PRACTICE_MODE", "0").strip().lower()
        in {"1", "true", "yes", "on"},
    }
