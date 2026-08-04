"""Automated live audio capture test. Detects devices, records 5 seconds, reports audio levels."""
import sys
import time
import wave
import io
sys.path.insert(0, '.')

from engine.audio_recorder import AudioRecorder
import numpy as np

print("=== Auto-detecting audio devices on your laptop ===")
mic_idx, sys_idx = AudioRecorder.auto_detect_devices()
print(f"Auto-detected: Mic={mic_idx}, System={sys_idx}\n")

if mic_idx < 0 and sys_idx < 0:
    print("❌ No devices detected. Cannot test.")
    sys.exit(1)

rec = AudioRecorder()
rec.set_devices(mic_idx, sys_idx)

print("=== Starting 5-second live capture ===")
rec.start_recording()

time.sleep(0.5)
mic_started = rec.mic_stream is not None
sys_started = rec.system_stream is not None
print(f"Mic stream started: {mic_started}")
print(f"System loopback stream started: {sys_started}\n")

if not mic_started and not sys_started:
    print("❌ Both streams failed. Live audio capture not working.")
    sys.exit(1)

print("Recording audio levels...")
for i in range(10):
    time.sleep(0.5)
    mic_rms = getattr(rec, 'mic_rms', 0)
    sys_rms = getattr(rec, 'system_rms', 0)
    mic_status = "🔊 AUDIO" if mic_rms > 0.001 else "SILENCE"
    sys_status = "🔊 AUDIO" if sys_rms > 0.001 else "SILENCE"
    print(f"  [{i+1}/10] Mic: {mic_rms:.5f} ({mic_status}) | System: {sys_rms:.5f} ({sys_status})")

print("\n=== Stopping capture ===")
rec.stop_recording()

print("\n✅ Live audio capture test completed.")
print(f"   Mic device index: {mic_idx}")
print(f"   System loopback device index: {sys_idx}")
print(f"   Mic stream active: {mic_started}")
print(f"   System stream active: {sys_started}")
