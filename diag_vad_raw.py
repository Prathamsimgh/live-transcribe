"""Feed REAL captured speech frames directly to VAD, bypassing the gate."""
import subprocess, threading, time
import numpy as np
from config import Settings
import vad as vad_mod
import audio

s = Settings()
v, desc = vad_mod.load_vad(s.target_rate)
print(f"VAD: {desc}")

frames = []
cap = audio.LoopbackCapture(on_frame=frames.append,
                            target_rate=s.target_rate, frame_samples=s.frame_samples,
                            on_error=lambda e: print("ERR", e))
cap.start(); time.sleep(0.3)

def speak():
    subprocess.run(["powershell", "-NoProfile", "-c",
        "Add-Type -AssemblyName System.Speech; "
        "$sp = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$sp.Volume = 100; $sp.Speak('Hello. This is a test of the voice activity detector.')"],
        capture_output=True)

t = threading.Thread(target=speak); t.start()
time.sleep(6); t.join(timeout=6)
cap.stop()

print(f"captured {len(frames)} frames")
print("VAD probabilities on real speech frames:")
for i, f in enumerate(frames[::8]):  # every 8th frame
    p = v(f.samples)
    print(f"  frame {i*8:4d}: rms={f.rms:.4f} vad_prob={p:.4f}")
