from PySide6.QtCore import QObject
from PySide6.QtMultimedia import QMediaDevices

import config


def camera_device_id(device):
    try:
        return bytes(device.id()).hex()
    except Exception:
        return ""


def list_cameras():
    cameras = []
    try:
        default = QMediaDevices.defaultVideoInput()
        default_id = camera_device_id(default)
        for device in QMediaDevices.videoInputs():
            cameras.append(
                {
                    "id": camera_device_id(device),
                    "name": device.description() or "Camera",
                    "default": camera_device_id(device) == default_id,
                }
            )
    except Exception as exc:
        print(f"[camera-monitor] Could not enumerate cameras: {exc}")
    return cameras


class CameraDeviceMonitor(QObject):
    """Track camera hot-plug events and keep live capture on the selected device."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.media_devices = QMediaDevices(self)
        self._last_ids = {item["id"] for item in list_cameras() if item["id"]}
        self.media_devices.videoInputsChanged.connect(self._refresh)
        self._refresh(initial=True)
        print("[camera-monitor] Camera hot-plug detection enabled.")

    def stop(self):
        try:
            self.media_devices.videoInputsChanged.disconnect(self._refresh)
        except Exception:
            pass

    def _apply_capture_device(self, selected):
        capture = getattr(self.window, "camera_capture", None)
        if capture is not None:
            capture.set_device(selected)

    def _refresh(self, initial=False):
        if not self.window.settings.get("auto_detect_camera_devices", True):
            return

        cameras = list_cameras()
        ids = {item["id"] for item in cameras if item["id"]}
        added_ids = ids - self._last_ids
        self._last_ids = ids

        current = str(self.window.settings.get("camera_device_id", "") or "")
        selected = current
        selected_name = ""

        if (
            self.window.settings.get("auto_switch_new_camera", True)
            and added_ids
        ):
            newest = next(
                (
                    item
                    for item in reversed(cameras)
                    if item["id"] in added_ids
                ),
                None,
            )
            if newest:
                selected = newest["id"]
                selected_name = newest["name"]
        elif current not in ids:
            default_camera = next(
                (item for item in cameras if item.get("default")),
                None,
            )
            fallback = default_camera or (cameras[0] if cameras else None)
            if fallback:
                selected = fallback["id"]
                selected_name = fallback["name"]
            else:
                selected = ""

        if selected != current:
            self.window.settings["camera_device_id"] = selected
            config.save_settings(self.window.settings)
            self._apply_capture_device(selected)
            if selected:
                print(
                    "[camera-monitor] Active camera switched automatically: "
                    f"{selected_name or selected}"
                )
            else:
                print("[camera-monitor] No camera currently connected.")
        else:
            self._apply_capture_device(selected)
            if initial and selected:
                camera = next(
                    (item for item in cameras if item["id"] == selected),
                    None,
                )
                print(
                    f"[camera-monitor] Active camera: "
                    f"{(camera or {}).get('name', selected)}"
                )
