def install_hybrid_provider_settings():
    """Add NVIDIA/hybrid choices to the existing provider combo without duplicating UI code."""
    from ui.settings_dialog import SettingsDialog

    if getattr(SettingsDialog, "_hybrid_provider_patch_installed", False):
        return

    original_load_devices = SettingsDialog._load_devices

    def load_devices(dialog):
        original_load_devices(dialog)
        combo = getattr(dialog, "provider_combo", None)
        if combo is None:
            return

        current = str(dialog.settings.get("ai_provider", "hybrid") or "hybrid")
        if combo.findData("hybrid") < 0:
            combo.insertItem(1, "Hybrid · Kimi K3 → local Qwen", "hybrid")
        if combo.findData("nvidia") < 0:
            combo.insertItem(2, "NVIDIA Kimi K3 only", "nvidia")

        selected = combo.findData(current)
        if selected >= 0:
            combo.setCurrentIndex(selected)

    SettingsDialog._load_devices = load_devices
    SettingsDialog._hybrid_provider_patch_installed = True
