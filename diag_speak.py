"""Capture WHILE speaking known text through the default output device."""
import subprocess, threading, time
import numpy as np
from config import Settings
import audio

s = Settings()
frames = []
cap = audio.LoopbackCapture(on_frame=frames.append,
                            target_rate=s.target_rate,
                            frame_samples=s.frame_samples,
                            on_error=lambda e: print("AUDIO ERROR:", e))
cap.start()
print(f"capture: {cap.device_name} @ {cap.device_rate}Hz ch={cap._channels}")
time.sleep(0.5)

def speak():
    subprocess.run(["powershell", "-NoProfile", "-c",
        "Add-Type -AssemblyName System.Speech; "
        "$sp = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$sp.Volume = 100; $sp.Rate = 0; "
        "$sp.Speak('The quick brown fox jumps over the lazy dog. "
        "Testing one two three. This is a speech recognition test.')"],
        capture_output=True)

print("frames before speech:", len(frames))
t = threading.Thread(target=speak); t.start()
for i in range(10):
    time.sleep(1)
    if frames:
        rms = np.mean([f.rms for f in frames[-31:]])
        print(f"  t+{i+1}s frames={len(frames)} recent_rms={rms:.4f}")
    else:
        print(f"  t+{i+1}s frames=0")
    if not t.is_alive() and frames: break
t.join(timeout=8)
time.sleep(1)
cap.stop()
print(f"TOTAL frames: {len(frames)} ({len(frames)*0.032:.1f}s of audio)")
if frames:
    all_rms = np.array([f.rms for f in frames])
    print(f"RMS mean={all_rms.mean():.4f} max={all_rms.max():.4f} nonzero={np.count_nonzero(all_rms)}/{len(all_rms)}")
