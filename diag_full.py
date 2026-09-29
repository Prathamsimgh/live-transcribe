"""Speak text -> capture -> VAD/segmenter -> ASR. Full pipeline on real audio."""
import subprocess, threading, time
import numpy as np
from config import Settings
import vad as vad_mod
from segmenter import Segmenter
import audio, asr

s = Settings()
v, desc = vad_mod.load_vad(s.target_rate)
print(f"VAD: {desc}")
seg = Segmenter(s, v)
utterances, probs = [], []

def on_frame(f):
    u = seg.push(f)
    if u is not None:
        utterances.append(u)

cap = audio.LoopbackCapture(on_frame=on_frame, target_rate=s.target_rate,
                            frame_samples=s.frame_samples,
                            on_error=lambda e: print("AUDIO ERROR:", e))
cap.start()
time.sleep(0.5)

def speak():
    subprocess.run(["powershell", "-NoProfile", "-c",
        "Add-Type -AssemblyName System.Speech; "
        "$sp = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$sp.Volume = 100; "
        "$sp.Speak('The first question is about hash maps. "
        "Then we discuss binary search trees.')"], capture_output=True)

t = threading.Thread(target=speak); t.start()
for i in range(8):
    time.sleep(1)
    print(f"  t+{i+1}s in_speech={seg._in_speech} buf={len(seg._buffer)} "
          f"floor={seg.noise_floor:.4f} utterances={len(utterances)}")
    if not t.is_alive(): time.sleep(1); break
tail = seg.flush()
if tail: utterances.append(tail)
cap.stop()

print(f"\nUTTERANCES: {len(utterances)}")
for i, u in enumerate(utterances):
    rms = float(np.sqrt(np.mean(u.samples.astype(np.float64)**2)))
    print(f"  [{i}] dur={u.duration_s:.2f}s mean_vad={u.mean_vad_prob:.3f} rms={rms:.4f}")

if utterances:
    print("\nLoading model and transcribing...")
    tr = asr.Transcriber(s)
    print(f"device={tr.device}")
    for i, u in enumerate(utterances):
        t0 = time.time()
        text = tr.transcribe(u)
        print(f"  [{i}] ({time.time()-t0:.2f}s) -> {text!r}")
