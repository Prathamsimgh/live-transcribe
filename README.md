# Live Transcribe

Real-time transcription of Windows system audio, with one-click toggle,
append-only daily transcript files, and hotkey clipboard copy.

Runs fully offline on the local GPU. See
`docs/superpowers/specs/2026-09-28-live-transcribe-design.md` for the design.

## Status

Implemented and verified end-to-end on the target machine: capture at 192 kHz,
Silero VAD segmentation, faster-whisper on CUDA, and transcript output all
confirmed working.

## Run it

Double-click `Live Transcribe.vbs`, or:

```
run_transcribe.bat
```

On first run it downloads the `small.en` Whisper model (~250 MB) and shows a
`downloading model...` status line while doing so.

## Setup from a fresh clone

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

`config.json` is created next to the script on first run, holding every tunable
in one place. Delete it to return to defaults.

## Tests

```
.venv\Scripts\python -m pytest tests/ -q
```

Hardware and the GPU are injected, so the suite runs without a sound device.

The `diag_*.py` scripts need real audio and drive the pipeline against whatever
the speakers are playing. `diag_full.py` speaks a known sentence and prints the
transcript, which is the quickest way to check the whole chain.
