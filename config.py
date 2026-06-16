import os
import json

CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".ai_interview_copilot_settings.json")

DEFAULT_SETTINGS = {
    "provider": "gemini",
    "model": "gemini-2.5-flash",
    "api_key": "",
    "mic_device_idx": -1,
    "system_device_idx": -1,
    "hotkey_capture": "<ctrl>+<shift>+s",
    "hotkey_record": "<ctrl>+<shift>+a",
    "capture_region": None,  # None means FULL SCREEN capture automatically!
    "window_opacity": 0.90,
    "invisible_mode": True,
    "font_size": 13,
    "always_on_top": True
}

def load_settings():
    """
    Loads user configuration settings from the local JSON file.
    If the file doesn't exist, returns default configurations.
    """
    if not os.path.exists(CONFIG_FILE):
        return DEFAULT_SETTINGS.copy()
    
    try:
        with open(CONFIG_FILE, "r") as f:
            user_data = json.load(f)
            
        # Merge default settings to ensure new parameters are populated
        settings = DEFAULT_SETTINGS.copy()
        for k, v in user_data.items():
            if k in settings:
                if isinstance(settings[k], dict) and isinstance(v, dict):
                    settings[k].update(v)
                else:
                    settings[k] = v
                    
        # Auto-migrate legacy/removed provider settings to gemini
        if settings.get("provider") in ["openrouter", "auto", "openai", "anthropic"]:
            settings["provider"] = "gemini"
            settings["model"] = "gemini-2.5-flash"
            
        return settings
    except Exception as e:
        print(f"[config] Error loading settings: {e}")
        return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    """
    Saves the user configuration settings dictionary to the local JSON file.
    """
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(settings, f, indent=4)
        print(f"[config] Settings saved successfully to {CONFIG_FILE}")
        return True
    except Exception as e:
        print(f"[config] Error saving settings: {e}")
        return False
