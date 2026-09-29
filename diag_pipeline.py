"""End-to-end live diagnostic: capture real system audio and print stats."""
import time, numpy as np
from config import Settings
import vad as vad_mod
from segmenter import Segmenter

s = Settings()
v, desc = vad_mod.load_vad(s.target_rate)
print(f"VAD: {desc}")

frames, utterances = [], []
seg = Segmenter(s, v)

def on_frame(f):
    frames.append(f)
    u = seg.push(f)
    if u is not None:
        utterances.append(u)
        print(f"  >>> UTTERANCE {u.duration_s:.2f}s mean_vad={u.mean_vad_prob:.3f} "
              f"rms={np.sqrt(np.mean(u.samples.astype(np.float64)**2)):.4f}")

import audio
cap = audio.LoopbackCapture(on_frame=on_frame, target_rate=s.target_rate,
                            frame_samples=s.frame_samples,
                            on_error=lambda e: print("AUDIO ERROR:", e),
                            on_device_change=lambda n: print("DEVICE CHANGED:", n))
cap.start()
print(f"capture: {cap.device_name} @ {cap.device_rate}Hz")
print("Listening 8 seconds — PLAY A VIDEO WITH SPEECH NOW")
t0 = time.time()
while time.time() - t0 < 8:
    time.sleep(1)
    recent = frames[-31:]
    if recent:
        rms = np.mean([f.rms for f in recent])
        print(f"  t+{int(time.time()-t0)}s frames={len(frames)} "
              f"rms_mean={rms:.4f} floor={seg.noise_floor:.4f} "
              f"in_speech={seg._in_speech} utterances={len(utterances)}")
tail = seg.flush()
if tail: utterances.append(tail); print("  >>> FLUSH UTTERANCE", tail.duration_s)
cap.stop()
print(f"\nTOTAL: {len(frames)} frames, {len(utterances)} utterances")
if utterances:
    import asr
    tr = asr.Transcriber(s)
    for i, u in enumerate(utterances[:3]):
        print(f"  [{i}] transcribing {u.duration_s:.2f}s ...")
        print("   ->", repr(tr.transcribe(u)))
elif frames:
    rms = np.array([f.rms for f in frames])
    print(f"RMS stats: mean={rms.mean():.5f} p90={np.percentile(rms,90):.5f} max={rms.max():.5f}")
    print("=> capture works, but no utterances formed (VAD never saw speech?)")
else:
    print("=> ZERO frames captured — audio callback never fired")
