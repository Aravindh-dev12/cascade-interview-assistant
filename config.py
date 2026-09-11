import json
import os

CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".ai_interview_copilot_settings.json")
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_LOCAL_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"

DEFAULT_SETTINGS = {
    "ai_provider": "hybrid",  # Kimi K3 first, local Ollama fallback
    "model": DEFAULT_GEMINI_MODEL,
    "local_model": DEFAULT_LOCAL_MODEL,
    "ollama_base_url": DEFAULT_OLLAMA_BASE_URL,
    "ollama_num_ctx": 8192,
    "mic_device_idx": -1,
    "system_device_idx": -1,
    "auto_detect_audio_devices": True,
    "auto_switch_new_microphone": True,
    "camera_device_id": "",
    "auto_detect_camera_devices": True,
    "auto_switch_new_camera": True,
    "camera_capture_enabled": True,
    "camera_frame_interval_ms": 450,
    "camera_context_max_age_seconds": 3.0,
    "include_camera_with_speech": True,
    "hotkey_capture": "<ctrl>+<shift>+s",
    "hotkey_record": "<ctrl>+<shift>+a",
    "capture_region": None,
    "auto_start_listening": True,
    "auto_answer_speech": True,
    "answer_cooldown_seconds": 0.25,
    "auto_screen_watch": True,
    "auto_answer_screen": True,
    "screen_watch_interval_ms": 650,
    "screen_stable_ms": 450,
    "screen_change_threshold": 0.055,
    "screen_context_max_age_seconds": 12.0,
    "include_screen_with_speech": True,
    "window_opacity": 0.94,
    "invisible_mode": False,
    "font_size": 13,
    "always_on_top": True,
}


def _normalize_provider(provider):
    provider = str(provider or "").strip().lower()
    return provider if provider in {"auto", "hybrid", "nvidia", "ollama", "gemini"} else "hybrid"


def _normalize_model(model):
    model = str(model or "").strip()
    return model if model.startswith("gemini-") else DEFAULT_GEMINI_MODEL


def _normalize_local_model(model):
    model = str(model or "").strip()
    return model or DEFAULT_LOCAL_MODEL


def load_settings():
    settings = DEFAULT_SETTINGS.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as file:
                user_data = json.load(file)
            for key, value in user_data.items():
                if key in settings:
                    settings[key] = value
        except Exception as exc:
            print(f"[config] Error loading settings: {exc}")

    settings["ai_provider"] = _normalize_provider(settings.get("ai_provider"))
    settings["model"] = _normalize_model(settings.get("model"))
    settings["local_model"] = _normalize_local_model(settings.get("local_model"))
    settings["ollama_base_url"] = str(
        settings.get("ollama_base_url") or DEFAULT_OLLAMA_BASE_URL
    ).rstrip("/")
    settings["ollama_num_ctx"] = max(2048, int(settings.get("ollama_num_ctx", 8192)))
    settings["camera_frame_interval_ms"] = max(
        200, int(settings.get("camera_frame_interval_ms", 450))
    )
    settings["camera_context_max_age_seconds"] = max(
        0.5, float(settings.get("camera_context_max_age_seconds", 3.0))
    )
    return settings


def save_settings(settings):
    """Persist runtime preferences only; API keys remain in project-local env files."""
    try:
        clean_settings = DEFAULT_SETTINGS.copy()
        for key in clean_settings:
            if key in settings:
                clean_settings[key] = settings[key]

        clean_settings["ai_provider"] = _normalize_provider(clean_settings.get("ai_provider"))
        clean_settings["model"] = _normalize_model(clean_settings.get("model"))
        clean_settings["local_model"] = _normalize_local_model(
            clean_settings.get("local_model")
        )
        clean_settings["ollama_base_url"] = str(
            clean_settings.get("ollama_base_url") or DEFAULT_OLLAMA_BASE_URL
        ).rstrip("/")
        clean_settings["ollama_num_ctx"] = max(
            2048, int(clean_settings.get("ollama_num_ctx", 8192))
        )
        clean_settings["camera_frame_interval_ms"] = max(
            200, int(clean_settings.get("camera_frame_interval_ms", 450))
        )
        clean_settings["camera_context_max_age_seconds"] = max(
            0.5, float(clean_settings.get("camera_context_max_age_seconds", 3.0))
        )

        with open(CONFIG_FILE, "w", encoding="utf-8") as file:
            json.dump(clean_settings, file, indent=4)
        print(f"[config] Settings saved successfully to {CONFIG_FILE}")
        return True
    except Exception as exc:
        print(f"[config] Error saving settings: {exc}")
        return False
