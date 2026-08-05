import os
import time

import numpy as np
from PySide6.QtCore import QThread, Signal

try:
    import riva.client
except ImportError:
    riva = None


class STTWorker(QThread):
    transcription_ready = Signal(str, str)
    status_updated = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, audio_recorder, api_key=None):
        super().__init__()
        self.audio_recorder = audio_recorder
        self.api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
        self.running = False

        # Voice activity detection / endpointing.
        self.silence_threshold = 0.005
        self.silence_duration_limit = 1.0
        self.min_speech_duration = 0.6
        self.max_speech_duration = 15.0

        self.mic_speech_buffer = []
        self.system_speech_buffer = []
        self.mic_speech_active = False
        self.mic_silence_start = None
        self.system_speech_active = False
        self.system_silence_start = None

        self.server = os.environ.get(
            "NVIDIA_RIVA_SERVER", "grpc.nvcf.nvidia.com:443"
        ).strip()
        self.function_id = os.environ.get(
            "NVIDIA_RIVA_FUNCTION_ID",
            "71203149-d3b7-4460-8231-1be2543a1fca",
        ).strip()
        self.language_code = os.environ.get("NVIDIA_RIVA_LANGUAGE", "en-US").strip()

        self._asr_service = None

    def set_api_key(self, api_key=None):
        """Refresh the NVIDIA key from the environment.

        The argument is intentionally ignored so Gemini/Ollama/OpenAI provider keys
        can never be accidentally used for speech-to-text.
        """
        self.api_key = os.environ.get("NVIDIA_API_KEY", "").strip()

    def stop(self):
        self.running = False
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
            ("function-id", self.function_id),
            ("authorization", f"Bearer {self.api_key}"),
        ]
        auth = riva.client.Auth(None, True, self.server, metadata)
        return riva.client.ASRService(auth)

    def run(self):
        self.running = True
        self.status_updated.emit("Parakeet listening...")

        try:
            self._asr_service = self._create_parakeet_client()
            print("[stt] NVIDIA Parakeet client initialized.")
        except Exception as exc:
            self._asr_service = None
            self.error_occurred.emit(f"Parakeet initialization error: {exc}")

        while self.running:
            if not self.audio_recorder.is_recording:
                time.sleep(0.05)
                continue

            mic_chunk, system_chunk = self.audio_recorder.get_next_audio_chunks()
            now = time.time()

            if mic_chunk is not None and len(mic_chunk) > 0:
                self._process_stream(
                    mic_chunk,
                    self.mic_speech_buffer,
                    self.mic_speech_active,
                    self.mic_silence_start,
                    "Candidate",
                    now,
                )

            if system_chunk is not None and len(system_chunk) > 0:
                self._process_stream(
                    system_chunk,
                    self.system_speech_buffer,
                    self.system_speech_active,
                    self.system_silence_start,
                    "Interviewer",
                    now,
                )

            time.sleep(0.05)

        print("[stt] Parakeet worker stopped.")

    def _process_stream(
        self,
        audio_chunk,
        speech_buffer,
        speech_active,
        silence_start,
        speaker_label,
        current_time,
    ):
        rms = np.sqrt(np.mean(audio_chunk ** 2)) if len(audio_chunk) else 0.0
        active_chunk = rms > self.silence_threshold
        is_mic = speaker_label == "Candidate"

        if active_chunk:
            if not speech_active:
                speech_active = True
                print(f"[stt] {speaker_label} started speaking (RMS: {rms:.4f})")
            speech_buffer.append(audio_chunk)
            silence_start = None
        elif speech_active:
            speech_buffer.append(audio_chunk)
            if silence_start is None:
                silence_start = current_time
            elif current_time - silence_start >= self.silence_duration_limit:
                self._finalize_and_transcribe(speech_buffer, speaker_label)
                speech_active = False
                silence_start = None

        total_samples = sum(len(chunk) for chunk in speech_buffer)
        duration = total_samples / self.audio_recorder.sample_rate
        if speech_active and duration >= self.max_speech_duration:
            self._finalize_and_transcribe(speech_buffer, speaker_label)
            speech_active = False
            silence_start = None

        if is_mic:
            self.mic_speech_active = speech_active
            self.mic_silence_start = silence_start
        else:
            self.system_speech_active = speech_active
            self.system_silence_start = silence_start

    def _finalize_and_transcribe(self, speech_buffer, speaker_label):
        if not speech_buffer:
            return

        full_audio = np.concatenate(speech_buffer)
        speech_buffer.clear()

        duration = len(full_audio) / self.audio_recorder.sample_rate
        if duration < self.min_speech_duration:
            return

        if self._asr_service is None:
            try:
                self.set_api_key()
                self._asr_service = self._create_parakeet_client()
            except Exception as exc:
                self.error_occurred.emit(f"Parakeet unavailable: {exc}")
                return

        try:
            # Riva LINEAR_PCM expects raw 16-bit mono PCM, not a WAV container.
            pcm16 = (
                np.clip(full_audio, -1.0, 1.0) * 32767.0
            ).astype(np.int16).tobytes()

            config = riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hertz=self.audio_recorder.sample_rate,
                language_code=self.language_code,
                max_alternatives=1,
                enable_automatic_punctuation=True,
            )

            print(f"[stt] Parakeet transcribing {speaker_label}: {duration:.1f}s")
            response = self._asr_service.offline_recognize(pcm16, config)

            parts = []
            for result in response.results:
                if result.alternatives:
                    text = result.alternatives[0].transcript.strip()
                    if text:
                        parts.append(text)

            transcription = " ".join(parts).strip()
            if transcription:
                print(f"[stt] {speaker_label}: {transcription}")
                self.transcription_ready.emit(speaker_label, transcription)
        except Exception as exc:
            error_msg = f"Parakeet STT Error ({speaker_label}): {exc}"
            print(f"[stt] {error_msg}")
            self.error_occurred.emit(error_msg)
