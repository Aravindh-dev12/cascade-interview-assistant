import os
import queue
import threading
import time

import numpy as np
from PySide6.QtCore import QThread, Signal

try:
    import riva.client
except ImportError:
    riva = None


class _ParakeetStreamingSession:
    """One low-latency Riva streaming ASR request for one spoken utterance."""

    _STOP = object()

    def __init__(self, asr_service, streaming_config, speaker_label, on_final, on_error):
        self.asr_service = asr_service
        self.streaming_config = streaming_config
        self.speaker_label = speaker_label
        self.on_final = on_final
        self.on_error = on_error

        self.audio_queue = queue.Queue(maxsize=128)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.closed = False
        self.last_partial = ""
        self.final_emitted = False

    def start(self):
        self.thread.start()

    def feed(self, pcm_bytes):
        if self.closed or not pcm_bytes:
            return
        try:
            self.audio_queue.put(pcm_bytes, timeout=0.1)
        except queue.Full:
            # Dropping a chunk is better than blocking the capture thread and
            # creating several seconds of accumulated latency.
            pass

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.audio_queue.put(self._STOP, timeout=0.2)
        except queue.Full:
            # Ensure the generator can terminate even if the network stalled.
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.audio_queue.put_nowait(self._STOP)
            except queue.Full:
                pass

    def _audio_chunks(self):
        while True:
            item = self.audio_queue.get()
            if item is self._STOP:
                return
            yield item

    def _run(self):
        try:
            responses = self.asr_service.streaming_response_generator(
                audio_chunks=self._audio_chunks(),
                streaming_config=self.streaming_config,
            )

            for response in responses:
                if not response.results:
                    continue

                for result in response.results:
                    if not result.alternatives:
                        continue

                    text = result.alternatives[0].transcript.strip()
                    if not text:
                        continue

                    if result.is_final:
                        if not self.final_emitted:
                            self.final_emitted = True
                            self.on_final(self.speaker_label, text)
                    else:
                        self.last_partial = text

            # Some hosted pipelines return the latest stable partial when the
            # client closes the stream instead of a final result. Preserve it.
            if not self.final_emitted and self.last_partial:
                self.final_emitted = True
                self.on_final(self.speaker_label, self.last_partial)

        except Exception as exc:
            self.on_error(f"Parakeet streaming error ({self.speaker_label}): {exc}")


class STTWorker(QThread):
    transcription_ready = Signal(str, str)
    status_updated = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, audio_recorder, api_key=None):
        super().__init__()
        self.audio_recorder = audio_recorder
        self.api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
        self.running = False

        # Local VAD is used only to decide when to close an utterance stream.
        # Riva is already processing the speech while it is being spoken.
        self.silence_threshold = float(os.environ.get("PARAKEET_VAD_THRESHOLD", "0.005"))
        self.silence_duration_limit = float(os.environ.get("PARAKEET_ENDPOINT_SECONDS", "0.56"))
        self.max_speech_duration = float(os.environ.get("PARAKEET_MAX_UTTERANCE_SECONDS", "20"))

        self.server = os.environ.get(
            "NVIDIA_RIVA_SERVER", "grpc.nvcf.nvidia.com:443"
        ).strip()
        self.function_id = os.environ.get(
            "NVIDIA_RIVA_FUNCTION_ID",
            "71203149-d3b7-4460-8231-1be2543a1fca",
        ).strip()
        self.language_code = os.environ.get("NVIDIA_RIVA_LANGUAGE", "en-US").strip()

        self._asr_service = None
        self._was_recording = False

        self.mic_speech_active = False
        self.mic_silence_start = None
        self.mic_speech_samples = 0
        self.mic_session = None

        self.system_speech_active = False
        self.system_silence_start = None
        self.system_speech_samples = 0
        self.system_session = None

    def set_api_key(self, api_key=None):
        """Refresh only NVIDIA credentials; other provider keys are ignored."""
        self.api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
        self._asr_service = None

    def stop(self):
        self.running = False
        self._close_all_sessions()
        self.wait()

    def _create_parakeet_client(self):
        if riva is None:
            raise RuntimeError(
                "nvidia-riva-client is not installed. Run: pip install -U nvidia-riva-client"
            )
        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY is not configured")
        if not self.function_id:
            raise RuntimeError("NVIDIA_RIVA_FUNCTION_ID is not configured")

        metadata = [
            ["function-id", self.function_id],
            ["authorization", f"Bearer {self.api_key}"],
        ]
        auth = riva.client.Auth(
            use_ssl=True,
            uri=self.server,
            metadata_args=metadata,
        )
        return riva.client.ASRService(auth)

    def _ensure_client(self):
        if self._asr_service is None:
            self.set_api_key()
            self._asr_service = self._create_parakeet_client()
            print("[stt] NVIDIA Parakeet streaming client initialized.")
        return self._asr_service

    def _build_streaming_config(self):
        recognition_config = riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=self.audio_recorder.sample_rate,
            language_code=self.language_code,
            max_alternatives=1,
            audio_channel_count=1,
            enable_automatic_punctuation=True,
        )
        streaming_config = riva.client.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=True,
        )

        # Keep server endpointing close to the local 560 ms target. The local
        # VAD also closes the stream, so this is an optimization rather than a
        # correctness dependency.
        endpoint_ms = max(300, int(self.silence_duration_limit * 1000))
        try:
            riva.client.add_endpoint_parameters_to_config(
                streaming_config,
                -1,
                -1.0,
                endpoint_ms,
                min(240, max(120, endpoint_ms // 2)),
                0.98,
                0.98,
            )
        except Exception:
            # Older Riva clients may not expose configurable endpointing.
            pass

        return streaming_config

    def _new_session(self, speaker_label):
        service = self._ensure_client()
        session = _ParakeetStreamingSession(
            service,
            self._build_streaming_config(),
            speaker_label,
            self._on_final_transcript,
            self._on_stream_error,
        )
        session.start()
        return session

    @staticmethod
    def _to_pcm16(audio_chunk):
        return (
            np.clip(audio_chunk, -1.0, 1.0) * 32767.0
        ).astype(np.int16).tobytes()

    def _on_final_transcript(self, speaker_label, text):
        text = text.strip()
        if text:
            print(f"[stt] {speaker_label}: {text}")
            self.transcription_ready.emit(speaker_label, text)

    def _on_stream_error(self, message):
        print(f"[stt] {message}")
        self.error_occurred.emit(message)
        # Recreate the shared gRPC client for the next utterance.
        self._asr_service = None

    def _close_all_sessions(self):
        if self.mic_session is not None:
            self.mic_session.close()
            self.mic_session = None
        if self.system_session is not None:
            self.system_session.close()
            self.system_session = None

        self.mic_speech_active = False
        self.mic_silence_start = None
        self.mic_speech_samples = 0
        self.system_speech_active = False
        self.system_silence_start = None
        self.system_speech_samples = 0

    def run(self):
        self.running = True
        self.status_updated.emit("Parakeet streaming ASR ready...")
        print("[stt] Parakeet streaming worker started.")

        while self.running:
            if not self.audio_recorder.is_recording:
                if self._was_recording:
                    self._close_all_sessions()
                    self._was_recording = False
                time.sleep(0.03)
                continue

            self._was_recording = True

            try:
                self._ensure_client()
            except Exception as exc:
                self.error_occurred.emit(f"Parakeet initialization error: {exc}")
                time.sleep(0.5)
                continue

            mic_chunk, system_chunk = self.audio_recorder.get_next_audio_chunks()
            now = time.monotonic()

            if mic_chunk is not None and len(mic_chunk) > 0:
                self._process_stream(mic_chunk, "Candidate", now)

            if system_chunk is not None and len(system_chunk) > 0:
                self._process_stream(system_chunk, "Interviewer", now)

            # 100 ms recorder chunks are polled faster than real time so the
            # queues stay nearly empty and network latency does not accumulate.
            time.sleep(0.02)

        self._close_all_sessions()
        print("[stt] Parakeet streaming worker stopped.")

    def _process_stream(self, audio_chunk, speaker_label, current_time):
        rms = np.sqrt(np.mean(audio_chunk ** 2)) if len(audio_chunk) else 0.0
        active_chunk = rms > self.silence_threshold
        pcm = self._to_pcm16(audio_chunk)
        is_mic = speaker_label == "Candidate"

        if is_mic:
            speech_active = self.mic_speech_active
            silence_start = self.mic_silence_start
            speech_samples = self.mic_speech_samples
            session = self.mic_session
        else:
            speech_active = self.system_speech_active
            silence_start = self.system_silence_start
            speech_samples = self.system_speech_samples
            session = self.system_session

        if active_chunk:
            if not speech_active:
                try:
                    session = self._new_session(speaker_label)
                except Exception as exc:
                    self._on_stream_error(f"Parakeet stream start error ({speaker_label}): {exc}")
                    return
                speech_active = True
                speech_samples = 0
                print(f"[stt] {speaker_label} started speaking (RMS: {rms:.4f})")

            session.feed(pcm)
            speech_samples += len(audio_chunk)
            silence_start = None

        elif speech_active and session is not None:
            # Send trailing silence too. Riva can use it for its own endpointing
            # while our local VAD measures the low-latency utterance boundary.
            session.feed(pcm)
            speech_samples += len(audio_chunk)

            if silence_start is None:
                silence_start = current_time
            elif current_time - silence_start >= self.silence_duration_limit:
                session.close()
                session = None
                speech_active = False
                silence_start = None
                speech_samples = 0

        duration = speech_samples / self.audio_recorder.sample_rate
        if speech_active and duration >= self.max_speech_duration:
            session.close()
            session = None
            speech_active = False
            silence_start = None
            speech_samples = 0

        if is_mic:
            self.mic_speech_active = speech_active
            self.mic_silence_start = silence_start
            self.mic_speech_samples = speech_samples
            self.mic_session = session
        else:
            self.system_speech_active = speech_active
            self.system_silence_start = silence_start
            self.system_speech_samples = speech_samples
            self.system_session = session
