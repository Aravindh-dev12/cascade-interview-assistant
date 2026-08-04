"""
Quick diagnostic script to test microphone and system loopback capture.
Run this while playing some audio on your laptop (YouTube, music, etc.)
"""
import sys
import time
import wave
import numpy as np
import sounddevice as sd

def test_devices():
    print("=" * 60)
    print("AUDIO DEVICE DIAGNOSTIC")
    print("=" * 60)

    # List all devices
    devices = sd.query_devices()
    host_apis = sd.query_hostapis()
    
    print("\n--- All Audio Devices ---")
    for idx, d in enumerate(devices):
        api_name = host_apis[d['hostapi']]['name']
        in_ch = d['max_input_channels']
        out_ch = d['max_output_channels']
        print(f"  [{idx}] {d['name']}")
        print(f"       API: {api_name} | IN: {in_ch} | OUT: {out_ch} | SR: {int(d['default_samplerate'])}")

    print("\n--- Default Devices ---")
    print(f"  Default Input:  {sd.default.device[0]}")
    print(f"  Default Output: {sd.default.device[1]}")

    # Check if WasapiSettings is available
    print(f"\n--- WASAPI Support ---")
    if hasattr(sd, 'WasapiSettings'):
        print("  WasapiSettings: YES (loopback should work)")
    else:
        print("  WasapiSettings: NO (loopback may fail!)")

    # Find WASAPI index
    wasapi_idx = None
    for idx, api in enumerate(host_apis):
        if "WASAPI" in api['name']:
            wasapi_idx = idx
            break
    print(f"  WASAPI API Index: {wasapi_idx}")

    # Suggest loopback device
    print("\n--- Suggested Loopback Device ---")
    loopback_candidates = []
    for idx, d in enumerate(devices):
        if wasapi_idx is not None and d['hostapi'] == wasapi_idx:
            if d['max_output_channels'] > 0:
                loopback_candidates.append(idx)
                print(f"  [{idx}] {d['name']} (WASAPI output - can use loopback)")
        if "loopback" in d['name'].lower() and d['max_input_channels'] > 0:
            print(f"  [{idx}] {d['name']} (explicit loopback input)")

    if not loopback_candidates:
        print("  No WASAPI output devices found!")
        return

    # Test the first candidate
    test_idx = loopback_candidates[0]
    dev_info = devices[test_idx]
    print(f"\n--- Testing device [{test_idx}]: {dev_info['name']} ---")

    sys_channels = min(2, dev_info['max_input_channels'] if dev_info['max_input_channels'] > 0 else dev_info['max_output_channels'])
    sys_samplerate = int(dev_info['default_samplerate'])
    duration = 5  # seconds
    chunks = []

    def callback(indata, frames, time_info, status):
        if status:
            print(f"    Stream status: {status}")
        chunks.append(indata.copy())

    stream_kwargs = dict(
        device=test_idx,
        channels=sys_channels,
        samplerate=sys_samplerate,
        blocksize=int(sys_samplerate * 0.5),
        callback=callback,
        dtype=np.float32
    )

    if sys.platform == "win32" and dev_info['max_output_channels'] > 0 and hasattr(sd, 'WasapiSettings'):
        stream_kwargs['extra_settings'] = sd.WasapiSettings(loopback=True)
        print("  Using WASAPI loopback=True")

    print(f"  Channels: {sys_channels}, Sample Rate: {sys_samplerate}")
    print(f"  Recording {duration}s... PLEASE PLAY SOME AUDIO NOW (music/YouTube)")

    try:
        with sd.InputStream(**stream_kwargs) as stream:
            time.sleep(duration)
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    if not chunks:
        print("  No audio captured! Loopback is not working.")
        return

    raw = np.concatenate(chunks, axis=0)
    if sys_channels > 1:
        raw = np.mean(raw, axis=1)
    else:
        raw = raw.flatten()

    rms = np.sqrt(np.mean(raw**2))
    peak = np.max(np.abs(raw))
    print(f"  Captured {len(raw)} samples")
    print(f"  RMS energy: {rms:.6f}")
    print(f"  Peak level: {peak:.6f}")

    if rms < 0.001:
        print("  WARNING: Audio is very quiet. Check system volume or if loopback is active.")
    else:
        print("  Audio looks good! Loopback is working.")

    # Save test file
    audio_int16 = (np.clip(raw, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open("test_loopback.wav", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sys_samplerate)
        w.writeframes(audio_int16.tobytes())
    print(f"  Saved test_loopback.wav - play it to verify!")

    # Also test mic
    print("\n--- Testing default microphone ---")
    mic_idx = sd.default.device[0]
    if mic_idx is not None and mic_idx >= 0:
        print(f"  Using mic [{mic_idx}]: {devices[mic_idx]['name']}")
        print(f"  Recording {duration}s... PLEASE SPEAK INTO YOUR MIC")
        mic_chunks = []
        def mic_callback(indata, frames, time_info, status):
            if status:
                print(f"    Mic status: {status}")
            mic_chunks.append(indata.copy())

        with sd.InputStream(device=mic_idx, channels=1, samplerate=16000, 
                           blocksize=8000, callback=mic_callback, dtype=np.float32) as s:
            time.sleep(duration)

        if mic_chunks:
            mic_raw = np.concatenate(mic_chunks).flatten()
            mic_rms = np.sqrt(np.mean(mic_raw**2))
            print(f"  Mic RMS: {mic_rms:.6f}")
            if mic_rms > 0.001:
                print("  Mic is working!")
            else:
                print("  Mic captured silence. Check mic selection/volume.")
        else:
            print("  No mic audio captured.")
    else:
        print("  No default mic found.")

if __name__ == "__main__":
    test_devices()
    print("\n" + "=" * 60)
    input("Press Enter to exit...")
