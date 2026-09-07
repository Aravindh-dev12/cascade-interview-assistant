import io
import os
import queue
import sys
import threading
import wave
from collections import deque

import numpy as np
import sounddevice as sd


class AudioRecorder:
    def __init__(self, sample_rate=16000, chunk_duration=0.1):
        self.sample_rate = int(sample_rate)
        self.chunk_duration = float(chunk_duration)
        self.chunk_size = int(self.sample_rate * self.chunk_duration)

        self.mic_device_idx = None
        self.system_device_idx = None
        self.mic_queue = queue.Queue(maxsize=64)
        self.system_queue = queue.Queue(maxsize=64)
        self.is_recording = False
        self.mic_stream = None
        self.system_stream = None
        self.sys_channels = 1
        self.sys_samplerate = self.sample_rate
        self.lock = threading.RLock()

        debug_seconds = max(0.0, float(os.environ.get("AUDIO_DEBUG_BUFFER_SECONDS", "0")))
        debug_chunks = max(1, int(debug_seconds / self.chunk_duration)) if debug_seconds else 1
        self.keep_debug_audio = debug_seconds > 0
        self.audio_buffer_mic = deque(maxlen=debug_chunks)
        self.audio_buffer_system = deque(maxlen=debug_chunks)

    @staticmethod
    def _put_latest(target_queue, chunk):
        try:
            target_queue.put_nowait(chunk)
            return
        except queue.Full:
            pass
        try:
            target_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            target_queue.put_nowait(chunk)
        except queue.Full:
            pass

    @staticmethod
    def list_devices():
        """Return microphones and actual input-capable loopback/system devices."""
        mics = []
        loopbacks = []
        try:
            devices = sd.query_devices()
            host_apis = sd.query_hostapis()
        except Exception as exc:
            print(f"[audio] Error querying devices: {exc}")
            return mics, loopbacks

        wasapi_idx = None
        if sys.platform == "win32":
            for idx, api in enumerate(host_apis):
                if "WASAPI" in api.get("name", ""):
                    wasapi_idx = idx
                    break

        for idx, device in enumerate(devices):
            name = device.get("name", f"Device {idx}")
            api_name = host_apis[device["hostapi"]].get("name", "Audio")
            if device.get("max_input_channels", 0) > 0:
                if "loopback" not in name.lower() and "stereo mix" not in name.lower():
                    mics.append({"index": idx, "name": name, "api": api_name})

                is_loopback = (
                    "loopback" in name.lower()
                    or "stereo mix" in name.lower()
                    or "what u hear" in name.lower()
                )
                if is_loopback and (wasapi_idx is None or device["hostapi"] == wasapi_idx or "stereo mix" in name.lower()):
                    loopbacks.append({"index": idx, "name": name, "api": api_name})

        if sys.platform != "win32" and not loopbacks:
            loopbacks.append({"index": -1, "name": "System loopback unavailable", "api": "Windows only"})
        return mics, loopbacks

    @staticmethod
    def auto_detect_devices():
        try:
            devices = sd.query_devices()
            mic_idx = sd.default.device[0]
            if mic_idx is None or mic_idx < 0:
                mic_idx = next(
                    (idx for idx, dev in enumerate(devices) if dev.get("max_input_channels", 0) > 0),
                    -1,
                )

            _, loopbacks = AudioRecorder.list_devices()
            system_idx = next((item["index"] for item in loopbacks if item.get("index", -1) >= 0), -1)
            if system_idx < 0:
                print("[audio] No input-capable system loopback found; mic-only mode available.")
            else:
                print(f"[audio] Auto-detected mic={mic_idx}, system={system_idx}")
            return int(mic_idx if mic_idx is not None else -1), int(system_idx)
        except Exception as exc:
            print(f"[audio] Auto-detect failed: {exc}")
            return -1, -1

    def set_devices(self, mic_idx, system_idx):
        with self.lock:
            self.mic_device_idx = int(mic_idx) if mic_idx is not None else -1
            self.system_device_idx = int(system_idx) if system_idx is not None else -1
            print(f"[audio] Devices configured - mic={self.mic_device_idx}, system={self.system_device_idx}")

    def _mic_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio] Mic stream status: {status}", file=sys.stderr)
        self._put_latest(self.mic_queue, indata.copy())

    def _system_callback(self, indata, frames, time_info, status):
        self._put_latest(self.system_queue, indata.copy())

    def _reset_queues(self):
        self.mic_queue = queue.Queue(maxsize=64)
        self.system_queue = queue.Queue(maxsize=64)
        self.audio_buffer_mic.clear()
        self.audio_buffer_system.clear()

    def start_recording(self):
        with self.lock:
            if self.is_recording:
                return
            self.is_recording = True
            self._reset_queues()

            if self.mic_device_idx is not None and self.mic_device_idx >= 0:
                try:
                    self.mic_stream = sd.InputStream(
                        device=self.mic_device_idx,
                        channels=1,
                        samplerate=self.sample_rate,
                        blocksize=self.chunk_size,
                        callback=self._mic_callback,
                        dtype=np.float32,
                        latency="low",
                    )
                    self.mic_stream.start()
                    print("[audio] Mic stream started.")
                except Exception as exc:
                    print(f"[audio] Failed to start mic stream: {exc}")
                    self.mic_stream = None

            if self.system_device_idx is not None and self.system_device_idx >= 0:
                try:
                    dev_info = sd.query_devices(self.system_device_idx)
                    max_inputs = int(dev_info.get("max_input_channels", 0))
                    if max_inputs <= 0:
                        raise RuntimeError("Selected system-audio device is not input-capable")
                    self.sys_channels = min(2, max_inputs)
                    self.sys_samplerate = int(dev_info.get("default_samplerate", self.sample_rate))
                    sys_blocksize = max(1, int(self.sys_samplerate * self.chunk_duration))
                    self.system_stream = sd.InputStream(
                        device=self.system_device_idx,
                        channels=self.sys_channels,
                        samplerate=self.sys_samplerate,
                        blocksize=sys_blocksize,
                        callback=self._system_callback,
                        dtype=np.float32,
                        latency="low",
                    )
                    self.system_stream.start()
                    print(
                        f"[audio] System input started ({self.sys_channels}ch @ {self.sys_samplerate} Hz)."
                    )
                except Exception as exc:
                    print(f"[audio] Failed to start system loopback: {exc}")
                    self.system_stream = None

            if self.mic_stream is None and self.system_stream is None:
                self.is_recording = False

    def stop_recording(self):
        with self.lock:
            self.is_recording = False
            for attr in ("mic_stream", "system_stream"):
                stream = getattr(self, attr)
                if stream is not None:
                    try:
                        stream.stop()
                        stream.close()
                    except Exception as exc:
                        print(f"[audio] Error closing {attr}: {exc}")
                    setattr(self, attr, None)
            print("[audio] Audio recording stopped.")

    @staticmethod
    def _drain(target_queue):
        chunks = []
        while True:
            try:
                chunks.append(target_queue.get_nowait())
            except queue.Empty:
                break
        return chunks

    def get_next_audio_chunks(self):
        mic_chunks = self._drain(self.mic_queue)
        system_chunks = self._drain(self.system_queue)

        mic_audio = None
        if mic_chunks:
            mic_audio = np.concatenate(mic_chunks, axis=0).astype(np.float32).flatten()
            if self.keep_debug_audio:
                self.audio_buffer_mic.append(mic_audio.copy())

        system_audio = None
        if system_chunks:
            raw = np.concatenate(system_chunks, axis=0)
            raw = np.mean(raw, axis=1) if raw.ndim > 1 and raw.shape[1] > 1 else raw.flatten()
            if self.sys_samplerate != self.sample_rate and len(raw) > 1:
                num_samples = max(1, int(len(raw) * self.sample_rate / self.sys_samplerate))
                system_audio = np.interp(
                    np.linspace(0, len(raw) - 1, num_samples),
                    np.arange(len(raw)),
                    raw,
                ).astype(np.float32)
            else:
                system_audio = raw.astype(np.float32)
            if self.keep_debug_audio:
                self.audio_buffer_system.append(system_audio.copy())

        return mic_audio, system_audio

    @staticmethod
    def save_to_wav_bytes(audio_data, sample_rate=16000) -> bytes:
        if audio_data is None or len(audio_data) == 0:
            return b""
        audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        return wav_io.getvalue()
