import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv


SUPPORTED_ENV_FILES = (".env", ".env.local", ".env.txt", "env")


def load_project_env(project_dir: Path):
    """Load local project secrets/config robustly without ever printing values.

    The first existing supported env file is loaded with override=True so stale or
    blank Windows environment variables cannot mask values from the project's .env.
    A few common aliases are normalized to the canonical names used by the app.
    """
    project_dir = Path(project_dir).resolve()
    selected = None

    for name in SUPPORTED_ENV_FILES:
        candidate = project_dir / name
        if candidate.is_file():
            selected = candidate
            break

    if selected is not None:
        # For this desktop app, the project-local env file is the intended source
        # of truth. This also fixes the case where Windows has an empty variable
        # with the same name already defined.
        load_dotenv(selected, override=True)
        parsed = dotenv_values(selected)
    else:
        parsed = {}

    def normalize(canonical, aliases):
        current = os.environ.get(canonical, "").strip()
        if current:
            return
        for alias in aliases:
            value = os.environ.get(alias, "").strip()
            if not value and parsed:
                value = str(parsed.get(alias, "") or "").strip()
            if value:
                os.environ[canonical] = value
                return

    normalize("GEMINI_API_KEY", ("GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY", "GEMINI_KEY"))
    normalize("NVIDIA_API_KEY", ("NVIDIA_KEY", "NVIDIA_APIKEY"))

    detected_names = []
    for name in (
        "NVIDIA_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GENAI_API_KEY",
        "PRACTICE_MODE",
    ):
        if os.environ.get(name, "").strip() or (parsed and parsed.get(name)):
            detected_names.append(name)

    return {
        "selected_path": selected,
        "exists": selected is not None,
        "detected_names": detected_names,
        "nvidia_loaded": bool(os.environ.get("NVIDIA_API_KEY", "").strip()),
        "gemini_loaded": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
        "practice_mode": os.environ.get("PRACTICE_MODE", "0").strip().lower()
        in {"1", "true", "yes", "on"},
    }
