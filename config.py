import json
import os

CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".ai_interview_copilot_settings.json")

DEFAULT_SETTINGS = {
    "provider": "gemini",
    "model": "gemini-3.6-flash",
    "api_key": "",
    "mic_device_idx": -1,
    "system_device_idx": -1,
    "hotkey_capture": "<ctrl>+<shift>+s",
    "hotkey_record": "<ctrl>+<shift>+a",
    "capture_region": None,
    "window_opacity": 0.90,
    "invisible_mode": True,
    "font_size": 13,
    "always_on_top": True,
}


def load_settings():
    if not os.path.exists(CONFIG_FILE):
        return DEFAULT_SETTINGS.copy()

    try:
        with open(CONFIG_FILE, "r") as file:
            user_data = json.load(file)

        settings = DEFAULT_SETTINGS.copy()
        for key, value in user_data.items():
            if key in settings:
                settings[key] = value

        # All legacy provider choices are migrated to Gemini-only answers.
        settings["provider"] = "gemini"
        if not str(settings.get("model", "")).startswith("gemini-"):
            settings["model"] = "gemini-3.6-flash"
        return settings
    except Exception as exc:
        print(f"[config] Error loading settings: {exc}")
        return DEFAULT_SETTINGS.copy()


def save_settings(settings):
    try:
        clean_settings = DEFAULT_SETTINGS.copy()
        for key in clean_settings:
            if key in settings:
                clean_settings[key] = settings[key]
        clean_settings["provider"] = "gemini"

        with open(CONFIG_FILE, "w") as file:
            json.dump(clean_settings, file, indent=4)
        print(f"[config] Settings saved successfully to {CONFIG_FILE}")
        return True
    except Exception as exc:
        print(f"[config] Error saving settings: {exc}")
        return False
