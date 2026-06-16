import time
import io
import os
import numpy as np
from PySide6.QtCore import QThread, Signal
from openai import OpenAI

class STTWorker(QThread):
    # Signals to communicate with the main UI thread
    # Format: (speaker_label, text)
    transcription_ready = Signal(str, str)
    status_updated = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, audio_recorder, api_key=None):
        super().__init__()
        self.audio_recorder = audio_recorder
        self.api_key = api_key
        self.running = False
        
        # Whisper STT config
        self.silence_threshold = 0.005  # Energy threshold (RMS) for voice activity
        self.silence_duration_limit = 1.2  # Seconds of silence before finalizing a phrase
        
        # Audio accumulator buffers for VAD
        self.mic_speech_buffer = []
        self.system_speech_buffer = []
        
        # Track active speech states
        self.mic_speech_active = False
        self.mic_silence_start = None
        
        self.system_speech_active = False
        self.system_silence_start = None
        
        # Minimum audio length to transcribe (seconds)
        self.min_speech_duration = 1.0
        
        # Maximum audio accumulation to prevent infinite buffers if background noise is constant
        self.max_speech_duration = 15.0

    def set_api_key(self, api_key):
        self.api_key = api_key

    def stop(self):
        self.running = False
        self.wait()

    def run(self):
        self.running = True
        self.status_updated.emit("STT Engine Listening...")
        
        # Initialize OpenAI Client
        client = None
        if self.api_key:
            try:
                client = OpenAI(api_key=self.api_key)
            except Exception as e:
                self.error_occurred.emit(f"Failed to init OpenAI client: {e}")
                
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
            time.sleep(0.1)
            
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
        
        # Run transcription in a separate thread/task or call it synchronously inside this worker loop
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
                model="whisper-1",
                file=wav_file,
                response_format="text"
            )
            
            transcription = response.strip()
            if transcription:
                print(f"[stt] {speaker_label}: {transcription}")
                self.transcription_ready.emit(speaker_label, transcription)
        except Exception as e:
            error_msg = f"Whisper STT Error ({speaker_label}): {e}"
            print(f"[stt] {error_msg}")
            self.error_occurred.emit(error_msg)
