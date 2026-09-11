import time

from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QObject, Qt, Signal
from PySide6.QtMultimedia import QCamera, QMediaCaptureSession, QMediaDevices, QVideoSink


def _camera_id(device) -> str:
    try:
        return bytes(device.id()).hex()
    except Exception:
        return ""


class CameraCaptureController(QObject):
    """Capture throttled live camera frames for multimodal AI context.

    The camera stays open while enabled. Frames are not sent anywhere by this class;
    it emits compressed JPEG bytes to the overlay bridge, which keeps only the latest
    frame and sends it to the selected AI provider only when a question needs vision.
    """

    frame_ready = Signal(bytes)
    status_updated = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, device_id="", interval_ms=450, max_width=960, parent=None):
        super().__init__(parent)
        self.device_id = str(device_id or "")
        self.interval_seconds = max(0.2, int(interval_ms) / 1000.0)
        self.max_width = max(480, int(max_width))
        self.enabled = False
        self.camera = None
        self._last_emit = 0.0

        self.capture_session = QMediaCaptureSession(self)
        self.video_sink = QVideoSink(self)
        self.capture_session.setVideoSink(self.video_sink)
        self.video_sink.videoFrameChanged.connect(self._on_video_frame)

    def configure(self, device_id=None, interval_ms=None):
        restart = False
        if device_id is not None and str(device_id or "") != self.device_id:
            self.device_id = str(device_id or "")
            restart = True
        if interval_ms is not None:
            self.interval_seconds = max(0.2, int(interval_ms) / 1000.0)

        if restart and self.enabled:
            self._open_camera()

    def set_device(self, device_id):
        self.configure(device_id=device_id)

    def start(self):
        if self.enabled and self.camera is not None:
            return
        self.enabled = True
        self._open_camera()

    def stop(self):
        if not self.enabled and self.camera is None:
            return
        self.enabled = False
        self._close_camera()

    def _select_device(self):
        cameras = list(QMediaDevices.videoInputs())
        if not cameras:
            return None

        if self.device_id:
            for device in cameras:
                if _camera_id(device) == self.device_id:
                    return device

        default = QMediaDevices.defaultVideoInput()
        if _camera_id(default):
            return default
        return cameras[0]

    def _close_camera(self):
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
            try:
                self.capture_session.setCamera(None)
            except Exception:
                pass
            self.camera.deleteLater()
            self.camera = None

    def _open_camera(self):
        self._close_camera()
        if not self.enabled:
            return

        device = self._select_device()
        if device is None:
            self.status_updated.emit("CAMERA OFF · no camera detected")
            return

        try:
            self.device_id = _camera_id(device)
            self.camera = QCamera(device, self)
            self.capture_session.setCamera(self.camera)
            try:
                self.camera.errorOccurred.connect(
                    lambda _error, message: self.error_occurred.emit(
                        f"Camera error: {message}"
                    )
                )
            except Exception:
                pass
            self.camera.start()
            self._last_emit = 0.0
            self.status_updated.emit(
                f"CAMERA LIVE · {device.description() or 'camera'}"
            )
            print(
                f"[camera] Live frame capture started: "
                f"{device.description() or self.device_id}"
            )
        except Exception as exc:
            self.error_occurred.emit(f"Could not start camera: {exc}")
            self._close_camera()

    def _encode_image(self, image):
        if image.isNull():
            return b""

        if image.width() > self.max_width:
            image = image.scaledToWidth(
                self.max_width,
                Qt.TransformationMode.SmoothTransformation,
            )

        data = QByteArray()
        buffer = QBuffer(data)
        if not buffer.open(QIODevice.OpenModeFlag.WriteOnly):
            return b""
        ok = image.save(buffer, "JPG", 78)
        buffer.close()
        if not ok:
            return b""
        return bytes(data)

    def _on_video_frame(self, frame):
        if not self.enabled or frame is None or not frame.isValid():
            return

        now = time.monotonic()
        if now - self._last_emit < self.interval_seconds:
            return

        try:
            image = frame.toImage()
            payload = self._encode_image(image)
            if not payload:
                return
            self._last_emit = now
            self.frame_ready.emit(payload)
        except Exception as exc:
            self.error_occurred.emit(f"Camera frame conversion failed: {exc}")
