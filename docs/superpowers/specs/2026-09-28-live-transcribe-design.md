# Live Transcribe — Design

**Date:** 2026-09-28
**Status:** Approved in brainstorming, pending spec review

## Goal

A one-click Windows app that transcribes whatever the PC's speakers are playing,
in near-real-time, appending finished sentences to a plain text file and copying
the full transcript to the clipboard on a hotkey.

Primary use: a video or call is playing; the user wants the questions and speech
captured to a shareable text block without typing.

## Non-goals

- Transcribing the user's own microphone (system audio only)
- Speaker diarisation ("who said what")
- Translation
- Word-by-word live streaming text (explicitly rejected — see Decision 5)
- Any cloud service or network call beyond the one-time model download

## Decisions

These were settled during brainstorming and are not open for reinterpretation
during implementation.

| # | Decision | Rationale |
|---|---|---|
| 1 | Local `faster-whisper` on the RTX 3050 | Offline, zero cost, and keeps the user's scarce 7.7GB system RAM free by using otherwise-idle VRAM |
| 2 | System audio only, via WASAPI loopback | Matches the stated use case; needs no virtual cable or driver install |
| 3 | Hotkey copies the full transcript; no auto-copy | Auto-copying every chunk would make the clipboard unusable for anything else |
| 4 | Small always-visible Tkinter window | User can see state and transcript; Tkinter ships with Python, so no extra dependency |
| 5 | Finished sentences only — no partial/hypothesis text | Whisper hallucinates on mid-word truncation, so partials visibly rewrite themselves. Removing them also enables cross-sentence context (see ASR below) and cuts GPU load ~70%. A VU meter replaces the "is it listening?" reassurance at zero GPU cost |
| 6 | GPU inference with automatic CPU fallback | ~1GB one-time cuDNN/cuBLAS download buys ~3x lower latency and avoids holding ~1.5GB of RAM |

## Verified environment

Measured on the target machine 2026-09-28, not assumed:

- ASUS TUF Gaming F17 FX706HCB, i5-11400H (6C/12T), RTX 3050 Laptop 4GB, 7.7GB
  usable RAM (single-channel), 87GB free on C:
- Python 3.11.9, NVIDIA driver 596.21
- **No CUDA toolkit and no cuDNN present.** `torch 2.6.0+cpu` is installed and is
  a CPU-only build, so it bundles no CUDA DLLs to borrow. The GPU path requires
  installing `nvidia-cudnn-cu12` and `nvidia-cublas-cu12` from pip.
- Already installed and reusable: `PyAudioWPatch 0.2.12.8`, `numpy 1.26.4`,
  `onnxruntime 1.22.1`, `pynput 1.8.2`, `av 15.0.0`, `tokenizers 0.22.0`
- **Default output device is `Speakers (Realtek(R) Audio)` running at 192000 Hz
  stereo**, exposed as loopback device index 13. The sample rate is unusually
  high; the implementation MUST read the device's rate at open time and MUST NOT
  hardcode 44100 or 48000.
- A second loopback device, `Speakers (Voice.ai Audio Cable)` at 48000 Hz, exists
  on this machine. Device selection must therefore match against the *current
  default output device* rather than picking the first loopback it finds.

## Architecture

Project root: `C:\Users\ASUS\live-transcribe\` — its own git repository. It must
NOT be committed into the `C:\Users\ASUS` home-directory repo.

```
live-transcribe/
  live_transcribe.py      entry point; wires threads and queues
  config.py               Settings dataclass, JSON load/save, defaults
  audio.py                WASAPI loopback capture -> 16kHz mono frames
  segmenter.py            VAD + utterance boundary detection
  asr.py                  faster-whisper wrapper; GPU with CPU fallback
  sinks.py                transcript file writer + clipboard
  ui.py                   Tkinter window, VU meter, global hotkeys
  config.json             user-editable settings (created on first run)
  requirements.txt
  Live Transcribe.vbs     one-click launcher (no console window)
  transcripts/            output, gitignored
  tests/
```

Each module has one job and is testable without audio hardware or a GPU. `audio`,
`asr`, and the clipboard are the only modules that touch the outside world, and
each is injectable so tests can substitute a fake.

### Threading model

Three threads, communicating only through queues and one atomic float. No shared
mutable state, no locks.

| Thread | Responsibility | Must never |
|---|---|---|
| Audio callback (owned by PortAudio) | Downmix, resample, push 512-sample frames to `frame_q`; publish RMS for the VU meter | Block, allocate unboundedly, or call into ASR |
| Worker | Drain `frame_q`, run VAD, accumulate utterances, call ASR, push results to `result_q` | Touch Tkinter widgets |
| Main / UI (Tkinter mainloop) | Poll `result_q` every 50ms, redraw VU meter every 33ms, handle button clicks | Perform ASR or file I/O inline |

`frame_q` is bounded. On overflow the *oldest* frames are dropped and a counter
increments, which surfaces as a `lagging` indicator. Dropping is correct here:
falling permanently behind live audio is worse than losing a moment of it.

## Data flow

```
Speakers (Realtek) 192000 Hz stereo float32
  |
  |  audio.py: downmix L+R -> mono, soxr streaming resample 192000 -> 16000
  v
16000 Hz mono float32, 512-sample frames (32.0 ms each)
  |
  |  segmenter.py: energy pre-gate -> Silero VAD -> utterance boundaries
  v
Utterance: contiguous float32 buffer, 0.32 s to 10.0 s
  |
  |  asr.py: faster-whisper small.en, int8_float16, on CUDA
  v
Text string
  |
  +--> sinks.py -> transcripts/YYYY-MM-DD.txt  (flushed + fsynced)
  +--> ui.py    -> appended to the scrolling transcript view
  +--> asr.py   -> retained as initial_prompt for the next utterance
```

512 samples at 16kHz is exactly the frame size Silero VAD v5 requires, so no
padding or re-chunking is needed between the resampler and the VAD.

## Segmentation algorithm

This module decides when a sentence has ended, and is the main determinant of
perceived latency. All thresholds live in `config.json`.

**Energy pre-gate.** Before calling VAD, compute frame RMS. If it is below
`noise_floor * 1.5`, classify as silence and skip the VAD call entirely. The
noise floor adapts as the 10th percentile of RMS over the trailing 5 seconds.
This keeps CPU near zero when nothing is playing, which is the common case.
Frames skipped by the gate are recorded as **VAD probability 0.0**, so the mean
VAD probability used by the hallucination guard stays well-defined.

RMS is computed once, in `audio.py`, on the 16kHz mono frame, and travels
attached to the frame. The VU meter and the energy gate therefore read the same
number and cannot diverge.

**Speech detection.** Silero VAD v5 via onnxruntime, threshold `0.5`.

**Boundaries.**

- *Start:* 3 consecutive speech frames (96ms). On start, prepend 8 frames
  (256ms) of pre-roll held in a ring buffer, so the first word is not clipped.
- *End:* 16 consecutive non-speech frames (512ms) finalises the utterance.
- *Minimum length:* utterances shorter than 320ms are discarded as noise (door
  clicks, notification sounds) and never reach the GPU.
- *Maximum length:* at 10.0s, force a cut — but cut at the **lowest-RMS frame
  within the trailing 1.5s** rather than at exactly 10.0s. Natural speech nearly
  always dips there, so monologues split at plausible boundaries instead of
  mid-word. Audio after the cut point remains buffered as the start of the next
  utterance and is not discarded.

  After a soft cut the segmenter stays in its **in-speech** state, with the
  retained tail as the new utterance buffer. It does NOT re-run start detection,
  because demanding 3 fresh speech frames would discard the retained audio and
  clip the word that straddled the cut.

## ASR configuration

`faster-whisper`, model `small.en`, `device="cuda"`, `compute_type="int8_float16"`
(~600MB VRAM, comfortable on a 4GB card).

- `beam_size=5` — affordable because each utterance is transcribed exactly once
- `language="en"`, `task="transcribe"`
- `vad_filter=False` — segmentation already happened upstream; re-running VAD
  would double-trim and clip words
- `condition_on_previous_text=True`, with `initial_prompt` set to the previous
  committed sentence truncated to 200 characters. This is what keeps names,
  jargon, and capitalisation consistent across sentences, and is only possible
  because partials were dropped.
- `temperature=[0.0, 0.2, 0.4]` — fallback sampling breaks repetition loops

**Hallucination guard.** Whisper reliably emits stock phrases on near-silence
("Thank you.", "Subtitles by the Amara.org community", a lone period). Drop a
result when any of these hold:

1. `avg_logprob < -1.0`
2. `no_speech_prob > 0.6`
3. The normalised text matches a configurable blocklist of known silence
   hallucinations, *and* the utterance's mean VAD probability was below 0.6

Condition 3 is deliberately conjunctive: "Thank you." is a legitimate thing for a
speaker to say, so it is only suppressed when the audio was also mostly silent.

### Loading CUDA DLLs on Windows

`ctranslate2` will not find pip-installed NVIDIA libraries on its own. Before
importing `faster_whisper` or `ctranslate2`, add the wheel `bin` directories to
the DLL search path via `os.add_dll_directory()`, resolving them from
`site.getsitepackages()`. Import order matters: this must run first.

Wrap model construction in a try/except. On any failure, retry with
`device="cpu"`, `compute_type="int8"`, and set a flag that renders a
`CPU mode (slower)` badge in the UI. The app must never fail to start because the
GPU stack is broken.

## Output format

One file per day, append-only: `transcripts\2026-09-28.txt`.

```
=== Session started 2026-09-28 14:32:10 ===
[14:32:15] So the first question is about hash maps.
[14:32:23] And then we move on to the next part.
=== Session stopped 2026-09-28 14:45:02 ===
```

- **UTF-8 with BOM, CRLF line endings** so Notepad renders it correctly
- `flush()` then `os.fsync()` after every committed line, so the file is complete
  up to the last sentence even if the process is killed
- Starting a new session on the same day appends a fresh `Session started`
  header rather than truncating
- Timestamps are wall-clock at the moment the utterance *ended*
- Closing the window while running stops capture and writes the
  `Session stopped` footer before the process exits, so no file is left without
  one

**Clipboard scope.** `Ctrl+Alt+C` and the Copy All button yield every sentence
transcribed **since the app process launched** — spanning multiple START/STOP
toggles, not just the most recent one. Sentences are joined by newlines with the
`===` markers and the `[HH:MM:SS]` timestamps stripped, because the intent is
pasting readable prose into a chat. Restarting the app resets this buffer; older
text stays available in the transcript file.

## UI

A small always-on-top resizable window:

```
+------------------------------------------+
| Live Transcribe            [CPU mode]    |
| [######## START ########]                |
| [>~~~~~~~~~~~~~~] listening...           |   <- VU meter, 33ms refresh
|------------------------------------------|
| [14:32:15] So the first question is...   |
| [14:32:23] And then we move on to...     |   <- autoscrolls
|------------------------------------------|
| [Copy All]  [Open in Notepad]            |
| transcripts\2026-09-28.txt               |
+------------------------------------------+
```

- The START button toggles to a red STOP when running. This is the single-click
  enable/disable.
- The VU meter is fed from the capture thread's RMS, independent of
  transcription. It distinguishes "no speech detected" from "wrong device".
- Status line shows `downloading model (250MB)...` on first run so startup does
  not look frozen.
- Global hotkeys via `pynput` (works unfocused, needs no admin):
  `Ctrl+Alt+C` copy all, `Ctrl+Alt+R` toggle start/stop.

Launcher: `Live Transcribe.vbs` invokes `pythonw.exe live_transcribe.py`,
producing no console window, and gets a desktop shortcut.

## Configuration

`config.json` is created next to the script on first run, holding every tunable
in one place. The defaults below are normative — the implementation must use
exactly these values.

| Key | Default | Meaning |
|---|---|---|
| `model` | `"small.en"` | faster-whisper model name |
| `device` | `"auto"` | `auto` tries CUDA then falls back to CPU; `cuda` or `cpu` force one |
| `compute_type_gpu` | `"int8_float16"` | GPU quantisation, ~600MB VRAM |
| `compute_type_cpu` | `"int8"` | CPU fallback quantisation |
| `beam_size` | `5` | Affordable since each utterance runs once |
| `target_rate` | `16000` | Hz; do not change, Silero requires it |
| `frame_samples` | `512` | 32.0ms at 16kHz; do not change, Silero requires it |
| `vad_threshold` | `0.5` | Silero speech probability cutoff |
| `energy_gate_multiplier` | `1.5` | Multiple of the adaptive noise floor |
| `speech_start_frames` | `3` | 96ms of speech opens an utterance |
| `speech_end_frames` | `16` | 512ms of silence closes it |
| `preroll_frames` | `8` | 256ms prepended so the first word is not clipped |
| `min_utterance_ms` | `320` | Shorter is discarded as noise |
| `max_utterance_s` | `10.0` | Forced soft cut point |
| `soft_cut_window_s` | `1.5` | Trailing window searched for the quietest frame |
| `max_logprob_reject` | `-1.0` | Hallucination guard condition 1 |
| `no_speech_reject` | `0.6` | Hallucination guard condition 2 |
| `hallucination_phrases` | see below | Guard condition 3 blocklist |
| `transcript_dir` | `"transcripts"` | Relative to the project root |
| `hotkey_copy` | `"ctrl+alt+c"` | Copy everything since launch |
| `hotkey_toggle` | `"ctrl+alt+r"` | Start/stop capture |
| `always_on_top` | `true` | Window stays above others |

Default `hallucination_phrases`: `"thank you"`, `"thanks for watching"`,
`"subtitles by the amara.org community"`, `"please subscribe"`, `"."`. Matched
case-insensitively after stripping punctuation, and only suppressed when mean VAD
probability was also below 0.6.

Unknown keys are ignored rather than fatal, so a config written by a future
version does not break the app.

## Failure modes

These are specified because each one is a realistic Windows failure, not a
hypothetical.

| Failure | Handling |
|---|---|
| User switches to headphones mid-session | WASAPI loopback dies when the default output device changes. Catch the stream error, re-enumerate devices, restart capture on the new default, and write a `=== audio device changed ===` marker into the transcript so the gap is visible |
| Default device becomes the Voice.ai cable | Device selection matches the current default output by name, so it follows the change rather than silently capturing a dead device |
| cuDNN/cuBLAS missing or mismatched | Fall back to CPU int8, show `CPU mode (slower)` badge, keep running |
| Model not yet downloaded | Show download status in the UI; do not block the Tkinter loop |
| GPU cannot keep up | Drop oldest queued frames, increment counter, show `lagging` indicator. Never drop a completed utterance |
| Nothing playing | Energy pre-gate rejects silence before VAD; idles near 0% CPU |
| Transcript file locked or unwritable | Buffer lines in memory, retry with backoff, surface a UI warning. Never lose a transcribed sentence to an I/O error |
| 192kHz device with an odd channel count | Downmix handles arbitrary channel counts by averaging, not by assuming stereo |

## Dependencies

To install:

```
faster-whisper        pulls ctranslate2, huggingface-hub
soxr                  high-quality 192000 -> 16000 resampling
silero-vad            VAD; runs on the already-present onnxruntime
pyperclip             clipboard
nvidia-cudnn-cu12     GPU, ~1GB combined with cuBLAS
nvidia-cublas-cu12
```

Already present, pinned in `requirements.txt` but not reinstalled:
`PyAudioWPatch`, `numpy`, `onnxruntime`, `pynput`.

Plus a ~250MB one-time `small.en` model download to the HuggingFace cache.

## Latency budget

Measured from the speaker falling silent to the line appearing:

| Stage | Cost |
|---|---|
| Silence confirmation (16 frames) | 512 ms |
| ASR, 3–5s utterance, small.en int8_float16 on RTX 3050 | 200–350 ms |
| Queue handoff + UI redraw | ~50 ms |
| **Total** | **~0.8–0.9 s** |

Resampling is sub-millisecond even at 12:1 and is not a factor. On the CPU
fallback the ASR stage becomes roughly 1.5–2.5s, giving ~2–3s total.

## Test strategy

TDD. Hardware and GPU are injected, so the whole suite runs on any machine
without a sound device.

| Module | Tests |
|---|---|
| `segmenter` | Synthetic speech/silence frame patterns assert: start after 3 speech frames, pre-roll is prepended, end after 16 silent frames, sub-320ms utterances discarded, 10s cut lands on the trailing minimum-RMS frame, post-cut audio is retained |
| `sinks` | Line format, session headers, same-day append does not truncate, BOM and CRLF present, fsync called per line, `PermissionError` buffers and retries, a `Session stopped` footer is written on shutdown, clipboard text spans multiple start/stop cycles and excludes headers and timestamps |
| `asr` | Mocked model asserts GPU→CPU fallback on construction failure, `vad_filter=False`, `initial_prompt` threading from the previous sentence, and each of the three hallucination-guard conditions independently |
| `audio` | Fake PyAudioWPatch asserts arbitrary-rate resampling to exactly 16kHz, N-channel downmix by averaging, 512-sample framing, callback never blocks, and device re-selection after a default-device change |
| `config` | Defaults load when `config.json` is absent; unknown keys are ignored rather than fatal |
| Integration (slow, opt-in) | A real WAV through the full pipeline asserts text reaches the transcript file. Marked slow because it needs the actual model |

## Open risks

- **cuDNN 9 / ctranslate2 compatibility.** `ctranslate2` 4.x requires cuDNN 9;
  the pip wheel provides it, but version drift is the likeliest install failure.
  The CPU fallback is the mitigation, so a failure here degrades rather than
  blocks.
- **192kHz capture** is an unusual configuration and the least-exercised path in
  PyAudioWPatch. If the stream proves unstable, the fallback is to set the
  Realtek device to 48kHz in Windows sound settings, which the implementation
  should mention in its error text rather than leaving the user guessing.
