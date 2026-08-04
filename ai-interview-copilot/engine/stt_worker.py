import time
import io
import os
import numpy as np
from PySide6.QtCore import QThread, Signal
from openai import OpenAI

# Optional local Whisper (open-source, no API key needed)
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None

# Optional Google Gemini for STT
try:
    from google import genai
    from google.genai import types as gtypes
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class STTWorker(QThread):
    # Signals to communicate with the main UI thread
    # Format: (speaker_label, text)
    transcription_ready = Signal(str, str)
    status_updated = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, audio_recorder, api_key=None, stt_provider="gemini", stt_model="base"):
        super().__init__()
        self.audio_recorder = audio_recorder
        self.api_key = api_key
        self.stt_provider = stt_provider
        self.stt_model = stt_model
        self.running = False
        self.local_model = None
        self.gemini_client = None
        
        # Whisper STT config
        self.silence_threshold = 0.005  # Energy threshold (RMS) for voice activity
        # Fast endpointing: begin transcription shortly after speech ends.
        self.silence_duration_limit = 0.25
        
        # Audio accumulator buffers for VAD
        self.mic_speech_buffer = []
        self.system_speech_buffer = []
        
        # Track active speech states
        self.mic_speech_active = False
        self.mic_silence_start = None
        
        self.system_speech_active = False
        self.system_silence_start = None
        
        # Minimum audio length to transcribe (seconds)
        self.min_speech_duration = 0.3
        
        # Maximum audio accumulation to prevent infinite buffers if background noise is constant
        self.max_speech_duration = 15.0

    def set_api_key(self, api_key):
        self.api_key = api_key
        # Re-initialize Gemini client if using gemini provider
        if self.stt_provider == "gemini" and api_key and GEMINI_AVAILABLE:
            try:
                self.gemini_client = genai.Client(api_key=api_key)
                print("[stt] Gemini STT client re-initialized with new key.")
            except Exception as e:
                print(f"[stt] Failed to re-init Gemini client: {e}")

    def set_stt_provider(self, provider, model="base"):
        old_provider = self.stt_provider
        self.stt_provider = provider
        self.stt_model = model
        # If switching to local, load the model immediately
        if provider == "local" and old_provider != "local":
            self._load_local_model()
        # If switching to gemini, init client
        if provider == "gemini" and old_provider != "gemini" and self.api_key and GEMINI_AVAILABLE:
            try:
                self.gemini_client = genai.Client(api_key=self.api_key)
                print("[stt] Gemini STT client initialized on provider switch.")
            except Exception as e:
                print(f"[stt] Failed to init Gemini client on switch: {e}")

    def _load_local_model(self):
        """Load local faster-whisper model."""
        if not FASTER_WHISPER_AVAILABLE:
            self.error_occurred.emit("Local STT selected but faster-whisper is not installed. Run: pip install faster-whisper")
            return
        try:
            device = "cuda" if self._has_cuda() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            print(f"[stt] Loading local Whisper model '{self.stt_model}' on {device} ({compute_type})...")
            self.local_model = WhisperModel(self.stt_model, device=device, compute_type=compute_type)
            print("[stt] Local Whisper model loaded successfully.")
        except Exception as e:
            self.error_occurred.emit(f"Failed to load local Whisper model: {e}")
            self.local_model = None

    @staticmethod
    def _has_cuda():
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def stop(self):
        self.running = False
        self.wait()

    def run(self):
        self.running = True
        self.status_updated.emit("STT Engine Listening...")
        
        # Initialize OpenAI Client (for API mode)
        client = None
        if self.stt_provider == "openai" and self.api_key:
            try:
                client = OpenAI(api_key=self.api_key)
            except Exception as e:
                self.error_occurred.emit(f"Failed to init OpenAI client: {e}")
        
        # Initialize Gemini Client (for Gemini STT mode)
        if self.stt_provider == "gemini":
            gemini_key = self.api_key or os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
            if gemini_key and GEMINI_AVAILABLE:
                try:
                    self.gemini_client = genai.Client(api_key=gemini_key)
                    print("[stt] Gemini STT client initialized.")
                except Exception as e:
                    self.error_occurred.emit(f"Failed to init Gemini client: {e}")
            elif not GEMINI_AVAILABLE:
                self.error_occurred.emit("Gemini STT selected but google-genai is not installed.")
            elif not gemini_key:
                self.error_occurred.emit("Gemini STT selected but no API key found.")
        
        # Initialize local Whisper model (for local mode)
        if self.stt_provider == "local":
            self._load_local_model()
                
        print("[stt] STT Worker thread started.")
        
        while self.running:
            if not self.audio_recorder.is_recording:
                time.sleep(0.1)
                continue
                
            # Fetch latest audio chunks from the recorder
            mic_chunk, system_chunk = self.audio_recorder.get_next_audio_chunks()
            
            # Use current timestamp for timing VAD silence duration
            current_time = time.time()
            
            # --- 1. Process Candidate (Microphone) Audio ---
            if mic_chunk is not None and len(mic_chunk) > 0:
                self._process_stream(
                    audio_chunk=mic_chunk,
                    speech_buffer=self.mic_speech_buffer,
                    speech_active=self.mic_speech_active,
                    silence_start=self.mic_silence_start,
                    speaker_label="Candidate",
                    client=client,
                    current_time=current_time
                )
                
            # --- 2. Process Interviewer (System Loopback) Audio ---
            if system_chunk is not None and len(system_chunk) > 0:
                self._process_stream(
                    audio_chunk=system_chunk,
                    speech_buffer=self.system_speech_buffer,
                    speech_active=self.system_speech_active,
                    silence_start=self.system_silence_start,
                    speaker_label="Interviewer",
                    client=client,
                    current_time=current_time
                )
                
            # Prevent high CPU usage
            time.sleep(0.05)
            
        print("[stt] STT Worker thread stopped.")

    def _process_stream(self, audio_chunk, speech_buffer, speech_active, silence_start, speaker_label, client, current_time):
        # Calculate RMS energy of the chunk
        rms = np.sqrt(np.mean(audio_chunk**2)) if len(audio_chunk) > 0 else 0
        is_active_chunk = rms > self.silence_threshold
        
        # Update references based on speaker to modify self attributes
        is_mic = (speaker_label == "Candidate")
        
        if is_active_chunk:
            if not speech_active:
                # Speech just started
                speech_active = True
                print(f"[stt] {speaker_label} started speaking (RMS: {rms:.4f})")
                
            # Append audio to buffer
            speech_buffer.append(audio_chunk)
            
            # Reset silence timer
            silence_start = None
        else:
            if speech_active:
                # Speech was active, now silent chunk
                speech_buffer.append(audio_chunk)
                
                if silence_start is None:
                    silence_start = current_time
                elif current_time - silence_start >= self.silence_duration_limit:
                    # Silence limit reached, finalize and transcribe!
                    self._finalize_and_transcribe(speech_buffer, speaker_label, client)
                    speech_active = False
                    silence_start = None
            else:
                # Just constant silence, ignore chunk to avoid giant silent files
                pass
                
        # Force finalize if speech buffer gets too long (e.g. constant background noise)
        total_samples = sum(len(c) for c in speech_buffer)
        duration = total_samples / self.audio_recorder.sample_rate
        if speech_active and duration >= self.max_speech_duration:
            print(f"[stt] {speaker_label} speech buffer reached max limit ({duration:.1f}s). Finalizing...")
            self._finalize_and_transcribe(speech_buffer, speaker_label, client)
            speech_active = False
            silence_start = None

        # Write back updated states to class variables
        if is_mic:
            self.mic_speech_active = speech_active
            self.mic_silence_start = silence_start
        else:
            self.system_speech_active = speech_active
            self.system_silence_start = silence_start

    def _finalize_and_transcribe(self, speech_buffer, speaker_label, client):
        if not speech_buffer:
            return
            
        # Combine all chunks
        full_audio = np.concatenate(speech_buffer)
        speech_buffer.clear()  # Clear for next phrase
        
        # Check if duration is too short to transcribe
        duration = len(full_audio) / self.audio_recorder.sample_rate
        if duration < self.min_speech_duration:
            return
            
        print(f"[stt] Transcribing {speaker_label} segment: {duration:.1f}s")
        
        # --- LOCAL WHISPER (Open-source, no API key) ---
        if self.stt_provider == "local" and self.local_model:
            try:
                segments, info = self.local_model.transcribe(
                    full_audio,
                    language="en",
                    task="transcribe",
                    vad_filter=False  # We already do VAD via silence detection
                )
                transcription = " ".join([seg.text.strip() for seg in segments]).strip()
                if transcription:
                    print(f"[stt] {speaker_label}: {transcription}")
                    self.transcription_ready.emit(speaker_label, transcription)
            except Exception as e:
                error_msg = f"Local Whisper STT Error ({speaker_label}): {e}"
                print(f"[stt] {error_msg}")
                self.error_occurred.emit(error_msg)
            return

        # --- GEMINI STT (uses Google Gemini API for transcription) ---
        if self.stt_provider == "gemini" and self.gemini_client:
            try:
                wav_bytes = self.audio_recorder.save_to_wav_bytes(full_audio, self.audio_recorder.sample_rate)
                # Gemini can process audio directly from bytes
                audio_part = gtypes.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=["Transcribe this audio exactly. Output only the spoken words, nothing else.", audio_part],
                    config=gtypes.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=1024
                    )
                )
                transcription = response.text.strip() if response.text else ""
                if transcription:
                    print(f"[stt] {speaker_label}: {transcription}")
                    self.transcription_ready.emit(speaker_label, transcription)
            except Exception as e:
                error_msg = f"Gemini STT Error ({speaker_label}): {e}"
                print(f"[stt] {error_msg}")
                self.error_occurred.emit(error_msg)
            return

        # --- OPENAI WHISPER API ---
        if not client or not self.api_key:
            # Simulation/Dry run if no API key provided
            simulated_text = f"<{speaker_label} speech detected - Configure OpenAI API Key to transcribe>"
            self.transcription_ready.emit(speaker_label, simulated_text)
            return
            
        try:
            # Convert float32 NumPy array to WAV bytes
            wav_bytes = self.audio_recorder.save_to_wav_bytes(full_audio, self.audio_recorder.sample_rate)
            
            # Convert bytes to file-like object with a name so the SDK registers the format
            wav_file = io.BytesIO(wav_bytes)
            wav_file.name = "audio.wav"
            
            # Send to Whisper API
            response = client.audio.transcriptions.create(
                model=os.environ.get(
                    "OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe"
                ),
                file=wav_file,
                response_format="text",
                language="en",
                prompt=(
                    "Transcribe only clearly spoken English words exactly. "
                    "Do not invent speech during silence, noise, music, or echo."
                ),
            )
            
            transcription = response.strip()
            if transcription:
                print(f"[stt] {speaker_label}: {transcription}")
                self.transcription_ready.emit(speaker_label, transcription)
        except Exception as e:
            error_msg = f"Whisper STT Error ({speaker_label}): {e}"
            print(f"[stt] {error_msg}")
            self.error_occurred.emit(error_msg)
