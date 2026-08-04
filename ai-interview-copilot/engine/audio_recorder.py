import queue
import threading
import time
import sys
import numpy as np
import sounddevice as sd
import wave
import io

try:
    import soundcard as sc
except ImportError:
    sc = None

class AudioRecorder:
    def __init__(self, sample_rate=16000, chunk_duration=0.3):
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
        self.system_thread = None
        self.system_stop_event = threading.Event()
        
        self.recording_thread = None
        
        # Audio storage (for debugging or exporting if needed)
        self.audio_buffer_mic = []
        self.audio_buffer_system = []
        
        # System loopback properties (set when stream starts)
        self.sys_channels = 2
        self.sys_samplerate = self.sample_rate

        # Mic stream properties (set when stream starts)
        self.mic_channels = 1
        self.mic_samplerate = self.sample_rate

        # Real-time audio levels for diagnostics
        self.mic_rms = 0.0
        self.system_rms = 0.0
        
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
        Handles Bluetooth earbuds, AirPods, USB headsets, and built-in speakers.
        Returns:
            tuple: (mic_idx, system_loopback_idx)
        """
        import sys
        try:
            devices = sd.query_devices()
            host_apis = sd.query_hostapis()
            
            # Find WASAPI host API index
            wasapi_idx = None
            for idx, api in enumerate(host_apis):
                if "WASAPI" in api['name']:
                    wasapi_idx = idx
                    break
            
            # 1. Find the best microphone — prefer WASAPI input, fallback to default
            mic_idx = -1
            
            # Try WASAPI default input first
            if wasapi_idx is not None:
                for idx, d in enumerate(devices):
                    if d['hostapi'] == wasapi_idx and d['max_input_channels'] > 0:
                        mic_idx = idx
                        break
            
            # Fallback to OS default input
            if mic_idx == -1:
                default_in = sd.default.device[0]
                if default_in is not None and default_in >= 0:
                    mic_idx = default_in
            
            # Last resort: first available input device
            if mic_idx == -1:
                for idx, d in enumerate(devices):
                    if d['max_input_channels'] > 0:
                        mic_idx = idx
                        break
            
            if mic_idx >= 0:
                print(f"[audio] Auto-detected mic: [{mic_idx}] {devices[mic_idx]['name']}")
            else:
                print("[audio] No microphone found!")
            
            # 2. Find the best system loopback device — MUST be WASAPI output for loopback to work
            system_idx = -1
            
            if wasapi_idx is not None:
                # Strategy A: Use the default WASAPI output device (what Windows is currently playing through)
                # This captures speakers, Bluetooth earbuds, AirPods — whatever is the default output
                try:
                    default_output = sd.default.device[1]
                    if default_output is not None and default_output >= 0:
                        dev = devices[default_output]
                        if dev['hostapi'] == wasapi_idx and dev['max_output_channels'] > 0:
                            system_idx = default_output
                            print(f"[audio] Using default WASAPI output for loopback: [{system_idx}] {dev['name']}")
                except Exception as e:
                    print(f"[audio] Error checking default output: {e}")
                
                # Strategy B: Scan ALL WASAPI output devices (covers Bluetooth/USB cases)
                if system_idx == -1:
                    for idx, d in enumerate(devices):
                        if d['hostapi'] == wasapi_idx and d['max_output_channels'] > 0:
                            system_idx = idx
                            print(f"[audio] Found WASAPI output device: [{idx}] {d['name']}")
                            break
            
            # Strategy C: Last resort — Stereo Mix (WDM-KS or DirectSound input)
            # Note: Stereo Mix is often disabled in Windows — WASAPI loopback is strongly preferred
            if system_idx == -1:
                for idx, d in enumerate(devices):
                    if d['max_input_channels'] > 0 and "stereo mix" in d['name'].lower():
                        system_idx = idx
                        print(f"[audio] Fallback: Stereo Mix found: [{idx}] {d['name']} (may be disabled in Windows)")
                        break
            
            if system_idx == -1:
                print("[audio] WARNING: No system loopback device found! Audio will be mic-only.")
            else:
                print(f"[audio] System loopback: [{system_idx}] {devices[system_idx]['name']}")
            
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
        self.mic_rms = float(np.sqrt(np.mean(indata**2)))
        self.mic_queue.put(indata.copy())

    def _system_callback(self, indata, frames, time_info, status):
        if status:
            # Print actual errors but skip common overflow/underflow noise
            status_str = str(status)
            if "overflow" not in status_str.lower() and "underflow" not in status_str.lower():
                print(f"[audio] System stream status: {status}", file=sys.stderr)
        self.system_rms = float(np.sqrt(np.mean(indata**2)))
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
                    dev_info = sd.query_devices(self.mic_device_idx)
                    mic_samplerate = int(dev_info['default_samplerate'])
                    mic_channels = min(1, dev_info['max_input_channels'])
                    mic_blocksize = int(mic_samplerate * self.chunk_duration)
                    
                    print(f"[audio] Opening mic [{self.mic_device_idx}] at {mic_samplerate}Hz, {mic_channels}ch")
                    
                    self.mic_stream = sd.InputStream(
                        device=self.mic_device_idx,
                        channels=mic_channels,
                        samplerate=mic_samplerate,
                        blocksize=mic_blocksize,
                        callback=self._mic_callback,
                        dtype=np.float32
                    )
                    self.mic_stream.start()
                    self.mic_samplerate = mic_samplerate
                    self.mic_channels = mic_channels
                    print(f"[audio] Mic stream started at {mic_samplerate}Hz.")
                except Exception as e:
                    print(f"[audio] Failed to start mic stream: {e}")
                    self.mic_stream = None
            
            # 2. Start System Loopback Stream (if configured)
            if self.system_device_idx is not None and self.system_device_idx >= 0:
                started = self._try_start_system_loopback(self.system_device_idx)
                # If primary device failed, try any other WASAPI output device
                if not started and sys.platform == "win32":
                    print("[audio] Primary loopback failed. Scanning for alternative WASAPI output devices...")
                    try:
                        devices = sd.query_devices()
                        host_apis = sd.query_hostapis()
                        wasapi_idx = None
                        for idx, api in enumerate(host_apis):
                            if "WASAPI" in api['name']:
                                wasapi_idx = idx
                                break
                        if wasapi_idx is not None:
                            for idx, d in enumerate(devices):
                                if idx == self.system_device_idx:
                                    continue
                                if d['hostapi'] == wasapi_idx and d['max_output_channels'] > 0:
                                    print(f"[audio] Trying alternative loopback device [{idx}]: {d['name']}")
                                    if self._try_start_system_loopback(idx):
                                        self.system_device_idx = idx  # Remember working device
                                        print(f"[audio] Fallback loopback device [{idx}] started successfully.")
                                        break
                    except Exception as scan_e:
                        print(f"[audio] Error scanning fallback loopback devices: {scan_e}")

    def _try_start_system_loopback(self, device_idx):
        """
        Attempts to start a system loopback stream on the given device index.
        Returns True on success, False on failure.
        """
        try:
            dev_info = sd.query_devices(device_idx)
            sys_channels = min(2, dev_info['max_input_channels'] if dev_info['max_input_channels'] > 0 else dev_info['max_output_channels'])
            sys_samplerate = int(dev_info['default_samplerate'])
            sys_blocksize = int(sys_samplerate * self.chunk_duration)

            print(f"[audio] Opening loopback [{device_idx}] with {sys_channels} channels, SR: {sys_samplerate}")

            # Sounddevice cannot open a normal Windows output endpoint as an
            # InputStream. SoundCard exposes the endpoint's WASAPI loopback.
            if sys.platform == "win32" and dev_info['max_output_channels'] > 0:
                if sc is None:
                    raise RuntimeError(
                        "Windows system-audio capture requires 'soundcard'. "
                        "Run: pip install soundcard"
                    )
                speaker_name = str(dev_info["name"])
                speaker = next(
                    (s for s in sc.all_speakers()
                     if speaker_name.lower() in s.name.lower()
                     or s.name.lower() in speaker_name.lower()),
                    sc.default_speaker(),
                )
                if speaker is None:
                    raise RuntimeError(f"No output endpoint matched '{speaker_name}'")
                loopback = sc.get_microphone(id=str(speaker.id), include_loopback=True)
                self.sys_channels = max(1, min(2, int(dev_info["max_output_channels"])))
                self.sys_samplerate = sys_samplerate
                self.system_stop_event.clear()
                self.system_stream = loopback
                self.system_thread = threading.Thread(
                    target=self._soundcard_loopback_worker,
                    args=(loopback, sys_blocksize),
                    name="wasapi-loopback",
                    daemon=True,
                )
                self.system_thread.start()
                print(f"[audio] [{device_idx}] WASAPI loopback started for {speaker.name}.")
                return True

            stream_kwargs = dict(
                device=device_idx,
                channels=sys_channels,
                samplerate=sys_samplerate,
                blocksize=sys_blocksize,
                callback=self._system_callback,
                dtype=np.float32
            )
            self.system_stream = sd.InputStream(**stream_kwargs)
            self.system_stream.start()
            self.sys_channels = sys_channels
            self.sys_samplerate = sys_samplerate
            print(f"[audio] [{device_idx}] System loopback stream started.")
            return True
        except Exception as e:
            print(f"[audio] [{device_idx}] Failed to start system loopback: {e}")
            self.system_stream = None
            return False

    def _soundcard_loopback_worker(self, loopback, blocksize):
        """Copy the selected Windows output endpoint continuously into the STT queue."""
        com_initialized = False
        try:
            # SoundCard uses Windows Core Audio COM objects. Every worker thread
            # that touches them must initialize COM itself.
            if sys.platform == "win32":
                import ctypes
                result = ctypes.windll.ole32.CoInitializeEx(None, 0)
                com_initialized = result in (0, 1)
            with loopback.recorder(
                samplerate=self.sys_samplerate,
                channels=self.sys_channels,
                blocksize=blocksize,
            ) as recorder:
                while self.is_recording and not self.system_stop_event.is_set():
                    data = recorder.record(numframes=blocksize)
                    if data is not None and len(data):
                        data = np.asarray(data, dtype=np.float32)
                        self._system_callback(data, len(data), None, None)
        except Exception as exc:
            print(f"[audio] WASAPI loopback worker failed: {exc}", file=sys.stderr)
        finally:
            self.system_stop_event.set()
            if com_initialized:
                ctypes.windll.ole32.CoUninitialize()

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
                    self.system_stop_event.set()
                    if hasattr(self.system_stream, "stop"):
                        self.system_stream.stop()
                    if hasattr(self.system_stream, "close"):
                        self.system_stream.close()
                except Exception as e:
                    print(f"[audio] Error stopping system loopback: {e}")
                self.system_stream = None
            if self.system_thread and self.system_thread.is_alive():
                self.system_thread.join(timeout=2.0)
            self.system_thread = None
                
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
            raw_mic = np.concatenate(mic_chunks, axis=0)
            if self.mic_channels > 1:
                raw_mic = np.mean(raw_mic, axis=1)
            else:
                raw_mic = raw_mic.flatten()
            
            # Downsample from mic_samplerate to 16000Hz if needed
            if self.mic_samplerate != self.sample_rate:
                num_samples = int(len(raw_mic) * self.sample_rate / self.mic_samplerate)
                mic_audio = np.interp(
                    np.linspace(0, len(raw_mic) - 1, num_samples),
                    np.arange(len(raw_mic)),
                    raw_mic
                ).astype(np.float32)
            else:
                mic_audio = raw_mic.astype(np.float32)
            
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
