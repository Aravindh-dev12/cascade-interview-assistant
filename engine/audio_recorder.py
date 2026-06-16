import queue
import threading
import time
import sys
import numpy as np
import sounddevice as sd
import wave
import io

class AudioRecorder:
    def __init__(self, sample_rate=16000, chunk_duration=0.5):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        self.mic_device_idx = None
        self.system_device_idx = None
        
        self.mic_queue = queue.Queue()
        self.system_queue = queue.Queue()
        
        self.is_recording = False
        self.mic_stream = None
        self.system_stream = None
        
        self.recording_thread = None
        
        # Audio storage (for debugging or exporting if needed)
        self.audio_buffer_mic = []
        self.audio_buffer_system = []
        
        # Lock for thread safety
        self.lock = threading.Lock()

    @staticmethod
    def list_devices():
        """
        Queries and returns lists of available microphones and system audio loopback devices.
        """
        mics = []
        loopbacks = []
        
        if sys.platform != "win32":
            # For MacOS/Linux fallback
            try:
                devices = sd.query_devices()
                for idx, d in enumerate(devices):
                    if d['max_input_channels'] > 0:
                        mics.append({"index": idx, "name": d['name']})
                return mics, [{"index": -1, "name": "System Loopback (Windows Only)"}]
            except Exception as e:
                print(f"[audio] Error listing devices: {e}")
                return [], []

        try:
            devices = sd.query_devices()
            host_apis = sd.query_hostapis()
            
            # Find WASAPI index
            wasapi_idx = None
            for idx, api in enumerate(host_apis):
                if "WASAPI" in api['name']:
                    wasapi_idx = idx
                    break
            
            for idx, d in enumerate(devices):
                # Standard input devices (microphones)
                if d['max_input_channels'] > 0:
                    # Exclude loopbacks from general microphones
                    if "loopback" not in d['name'].lower():
                        # Prefer WASAPI for lower latency, but list others too
                        mics.append({
                            "index": idx, 
                            "name": d['name'], 
                            "api": host_apis[d['hostapi']]['name']
                        })
                
                # Check for WASAPI loopback devices
                # On Windows WASAPI, output devices can be opened in loopback mode
                if wasapi_idx is not None and d['hostapi'] == wasapi_idx:
                    if d['max_input_channels'] > 0 and "loopback" in d['name'].lower():
                        loopbacks.append({
                            "index": idx,
                            "name": d['name'],
                            "api": "WASAPI"
                        })
                    elif d['max_output_channels'] > 0:
                        # On WASAPI, we can record from any output device in loopback mode
                        loopbacks.append({
                            "index": idx,
                            "name": f"[Loopback] {d['name']}",
                            "api": "WASAPI (Loopback)"
                        })
            
            # Fallback to "Stereo Mix" if no WASAPI loopback was found
            if not loopbacks:
                for idx, d in enumerate(devices):
                    if d['max_input_channels'] > 0 and "stereo mix" in d['name'].lower():
                        loopbacks.append({
                            "index": idx,
                            "name": d['name'],
                            "api": host_apis[d['hostapi']]['name']
                        })
                        
        except Exception as e:
            print(f"[audio] Error querying devices: {e}")
            
        return mics, loopbacks

    @staticmethod
    def auto_detect_devices():
        """
        Dynamically auto-detects the default microphone and Windows WASAPI system loopback.
        Returns:
            tuple: (mic_idx, system_loopback_idx)
        """
        import sys
        try:
            devices = sd.query_devices()
            
            # 1. Get the default OS microphone (returns input device index)
            # Default input device is sd.default.device[0]
            mic_idx = sd.default.device[0]
            
            # Check if default input exists, otherwise pick first available input
            if mic_idx is None or mic_idx < 0:
                for idx, d in enumerate(devices):
                    if d['max_input_channels'] > 0:
                        mic_idx = idx
                        break
                        
            # 2. Get the default Windows WASAPI system audio loopback index
            system_idx = -1
            
            host_apis = sd.query_hostapis()
            wasapi_idx = None
            for idx, api in enumerate(host_apis):
                if "WASAPI" in api['name']:
                    wasapi_idx = idx
                    break
                    
            if wasapi_idx is not None:
                # Only use explicit loopback input devices to avoid channel errors
                for idx, d in enumerate(devices):
                    if d['hostapi'] == wasapi_idx and d['max_input_channels'] > 0:
                        if "loopback" in d['name'].lower():
                            system_idx = idx
                            break
                                
            # Fallback: Find Stereo Mix (only if no WASAPI loopback found)
            if system_idx == -1:
                for idx, d in enumerate(devices):
                    if d['max_input_channels'] > 0 and "stereo mix" in d['name'].lower():
                        system_idx = idx
                        break
            
            # If still no system device, return -1 to disable it (mic-only mode)
            if system_idx == -1:
                print(f"[audio] No valid loopback device found. Running in mic-only mode.")
            else:
                print(f"[audio] Auto-detected active hardware - Mic Index: {mic_idx}, Loopback Speaker Index: {system_idx}")
            return mic_idx, system_idx
        except Exception as e:
            print(f"[audio] Critical error auto-detecting hardware: {e}")
            return -1, -1

    def set_devices(self, mic_idx, system_idx):
        """
        Sets the device indices to record from.
        """
        with self.lock:
            self.mic_device_idx = mic_idx
            self.system_device_idx = system_idx
            print(f"[audio] Devices configured - Mic: {mic_idx}, System: {system_idx}")

    def _mic_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio] Mic stream status: {status}", file=sys.stderr)
        self.mic_queue.put(indata.copy())

    def _system_callback(self, indata, frames, time_info, status):
        if status:
            # We filter out overflow/underflow warnings for loopback as they can be noisy
            pass
        self.system_queue.put(indata.copy())

    def start_recording(self):
        """
        Starts recording in background threads.
        """
        with self.lock:
            if self.is_recording:
                return
            
            self.is_recording = True
            
            # Clear previous queues
            self.mic_queue = queue.Queue()
            self.system_queue = queue.Queue()
            
            self.audio_buffer_mic = []
            self.audio_buffer_system = []
            
            # 1. Start Mic Stream (if configured)
            if self.mic_device_idx is not None and self.mic_device_idx >= 0:
                try:
                    self.mic_stream = sd.InputStream(
                        device=self.mic_device_idx,
                        channels=1,
                        samplerate=self.sample_rate,
                        blocksize=self.chunk_size,
                        callback=self._mic_callback,
                        dtype=np.float32
                    )
                    self.mic_stream.start()
                    print("[audio] Mic stream started.")
                except Exception as e:
                    print(f"[audio] Failed to start mic stream: {e}")
                    self.mic_stream = None
            
            # 2. Start System Loopback Stream (if configured)
            if self.system_device_idx is not None and self.system_device_idx >= 0:
                try:
                    # In sounddevice, Windows WASAPI loopback requires matching the output device's channels/samplerate
                    # Let's get the device info to get safe default parameters
                    dev_info = sd.query_devices(self.system_device_idx)
                    sys_channels = min(2, dev_info['max_input_channels'] if dev_info['max_input_channels'] > 0 else dev_info['max_output_channels'])
                    sys_samplerate = int(dev_info['default_samplerate'])
                    sys_blocksize = int(sys_samplerate * self.chunk_duration)
                    
                    print(f"[audio] Opening loopback with {sys_channels} channels, SR: {sys_samplerate}")
                    
                    self.system_stream = sd.InputStream(
                        device=self.system_device_idx,
                        channels=sys_channels,
                        samplerate=sys_samplerate,
                        blocksize=sys_blocksize,
                        callback=self._system_callback,
                        dtype=np.float32
                    )
                    self.system_stream.start()
                    # Keep track of custom loopback settings for downsampling later
                    self.sys_channels = sys_channels
                    self.sys_samplerate = sys_samplerate
                    print("[audio] System loopback stream started.")
                except Exception as e:
                    print(f"[audio] Failed to start system loopback: {e}")
                    self.system_stream = None

    def stop_recording(self):
        """
        Stops active streams and saves recording if desired.
        """
        with self.lock:
            if not self.is_recording:
                return
            
            self.is_recording = False
            
            if self.mic_stream:
                try:
                    self.mic_stream.stop()
                    self.mic_stream.close()
                except Exception as e:
                    print(f"[audio] Error stopping mic stream: {e}")
                self.mic_stream = None
                
            if self.system_stream:
                try:
                    self.system_stream.stop()
                    self.system_stream.close()
                except Exception as e:
                    print(f"[audio] Error stopping system loopback: {e}")
                self.system_stream = None
                
            print("[audio] Audio recording stopped.")

    def get_next_audio_chunks(self):
        """
        Retrieves accumulated audio data from both streams, downsamples if necessary,
        and mixes/returns them.
        
        Returns:
            tuple: (mic_audio_np, system_audio_np) - both float32 16kHz mono or None
        """
        mic_chunks = []
        while not self.mic_queue.empty():
            try:
                mic_chunks.append(self.mic_queue.get_nowait())
            except queue.Empty:
                break
                
        system_chunks = []
        while not self.system_queue.empty():
            try:
                system_chunks.append(self.system_queue.get_nowait())
            except queue.Empty:
                break

        mic_audio = None
        if mic_chunks:
            mic_audio = np.concatenate(mic_chunks, axis=0).flatten()
            self.audio_buffer_mic.append(mic_audio)

        system_audio = None
        if system_chunks:
            # System loopback might have multiple channels and custom samplerate
            # Let's average channels to mono and downsample to 16000Hz for Whisper
            raw_sys = np.concatenate(system_chunks, axis=0)
            if self.sys_channels > 1:
                # Average channels
                raw_sys = np.mean(raw_sys, axis=1)
            else:
                raw_sys = raw_sys.flatten()
                
            # Downsample from sys_samplerate to 16000Hz
            if self.sys_samplerate != self.sample_rate:
                # Simple linear interpolation downsampling
                num_samples = int(len(raw_sys) * self.sample_rate / self.sys_samplerate)
                system_audio = np.interp(
                    np.linspace(0, len(raw_sys) - 1, num_samples),
                    np.arange(len(raw_sys)),
                    raw_sys
                ).astype(np.float32)
            else:
                system_audio = raw_sys.astype(np.float32)
                
            self.audio_buffer_system.append(system_audio)

        return mic_audio, system_audio

    @staticmethod
    def save_to_wav_bytes(audio_data, sample_rate=16000) -> bytes:
        """
        Converts float32 numpy array audio to 16-bit PCM WAV bytes in-memory.
        """
        if audio_data is None or len(audio_data) == 0:
            return b""
        
        # Convert float32 to int16
        # Clip values to [-1.0, 1.0] to avoid wrap-around distortion
        audio_clipped = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)
        
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)      # Mono
            wav_file.setsampwidth(2)      # 2 bytes for 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
            
        return wav_io.getvalue()
