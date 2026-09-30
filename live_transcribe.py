"""Live Transcribe -- entry point.

Wires three threads: PortAudio's capture callback, a worker running VAD and
ASR, and the Tkinter main loop. They communicate only through queues, so
there are no locks in the hot path.
"""

from __future__ import annotations

import queue
import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.json"

from config import Settings  # noqa: E402
import audio  # noqa: E402
import sinks  # noqa: E402
import vad as vad_mod  # noqa: E402
from segmenter import Segmenter  # noqa: E402

# Bounded so a stalled GPU cannot grow memory without limit. On overflow the
# oldest frames go: falling permanently behind live audio is worse than
# losing a moment of it.
FRAME_QUEUE_MAX = 400  # ~12.8 s of audio

# Completed utterances waiting for the GPU. Bounded for memory, but the
# segmenter blocks when full rather than dropping one (spec: never drop a
# completed utterance) -- backpressure then surfaces as frame drops instead.
UTTERANCE_QUEUE_MAX = 8


def db_level(rms: float) -> float:
    """Map RMS to a 0..1 bar over a -60..0 dB range."""
    if rms <= 1e-6:
        return 0.0
    db = 20.0 * np.log10(rms)
    return float(max(0.0, min(1.0, (db + 60.0) / 60.0)))


class App:
    def __init__(self) -> None:
        self.s = Settings.load(CONFIG_PATH)
        if not CONFIG_PATH.exists():
            self.s.save(CONFIG_PATH)

        self.writer = sinks.TranscriptWriter(ROOT / self.s.transcript_dir)
        self.buffer = sinks.SessionBuffer()

        self.frame_q: "queue.Queue" = queue.Queue(maxsize=FRAME_QUEUE_MAX)
        self.capture = audio.LoopbackCapture(
            on_frame=self._on_frame,
            target_rate=self.s.target_rate,
            frame_samples=self.s.frame_samples,
            on_device_change=self._on_device_change,
            on_error=self._on_audio_error,
        )

        self.vad, self.vad_desc = vad_mod.load_vad(self.s.target_rate)
        self.segmenter = Segmenter(self.s, self.vad)
        self.transcriber = None

        self.running = False
        self._worker: threading.Thread | None = None
        self._asr_worker: threading.Thread | None = None
        self.utterance_q: "queue.Queue" = queue.Queue(maxsize=UTTERANCE_QUEUE_MAX)
        self._dropped = 0
        self.window = None

    # ----------------------------------------------------------- UI binding

    def attach(self, window) -> None:
        self.window = window
        window.post("path", str(self.writer.path))
        window.post("status", "idle")

    def _say(self, kind: str, payload=None) -> None:
        if self.window is not None:
            self.window.post(kind, payload)

    # -------------------------------------------------------- audio thread

    def _on_frame(self, frame) -> None:
        """PortAudio callback thread. Must stay cheap."""
        if self.window is not None:
            self.window.set_level(db_level(frame.rms))
        if not self.running:
            return
        try:
            self.frame_q.put_nowait(frame)
        except queue.Full:
            try:
                self.frame_q.get_nowait()  # drop oldest
                self.frame_q.put_nowait(frame)
                self._dropped += 1
            except queue.Empty:
                pass

    def _on_device_change(self, name: str) -> None:
        self.writer.note(f"audio device changed to {name}")
        self._say("note", f"audio device changed to {name}")
        self._say("badge", self._badge())

    def _on_audio_error(self, exc: Exception) -> None:
        self._say("status", f"audio error: {type(exc).__name__}")

    # ------------------------------------------------------- worker threads

    def _work(self) -> None:
        """Segmentation thread: frames -> utterances. Never blocks on ASR,
        so VAD boundaries stay aligned to live audio while the GPU is busy.
        """
        while self.running:
            try:
                frame = self.frame_q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                utterance = self.segmenter.push(frame)
            except Exception:
                continue
            if utterance is not None:
                self._emit(utterance)
            if self._dropped:
                self._say("status", f"lagging ({self._dropped} dropped)")
                self._dropped = 0

        tail = self.segmenter.flush()
        if tail is not None:
            self._emit(tail)
        # Sentinel: the ASR thread exits only after draining every queued
        # utterance, so STOP never loses the final sentence.
        self._enqueue(None)

    def _enqueue(self, item) -> None:
        while True:
            try:
                self.utterance_q.put(item, timeout=0.25)
                return
            except queue.Full:
                continue

    def _emit(self, utterance) -> None:
        self._enqueue(utterance)

    def _asr_work(self) -> None:
        """ASR thread: utterances -> text, in order, one GPU call at a time."""
        while True:
            utterance = self.utterance_q.get()
            if utterance is None:
                return
            self._process(utterance)

    def _process(self, utterance) -> None:
        if self.transcriber is None:
            return
        try:
            text = self.transcriber.transcribe(utterance)
        except Exception as exc:
            self._say("status", f"asr error: {type(exc).__name__}")
            return
        if not text:
            return
        when = datetime.now()
        self.writer.write_line(text, when)
        self.buffer.add(text)
        self._say("line", (when.strftime("%H:%M:%S"), text))
        self._say("status", "listening")

    # ------------------------------------------------------------- controls

    def _badge(self) -> str:
        bits = []
        if self.transcriber is not None:
            bits.append("GPU" if self.transcriber.device == "cuda"
                        else "CPU mode (slower)")
        if self.capture.device_rate:
            bits.append(f"{self.capture.device_rate // 1000}kHz")
        return "  ".join(bits)

    def toggle(self) -> None:
        if self.running:
            self.stop()
        else:
            threading.Thread(target=self.start, daemon=True).start()

    def start(self) -> None:
        if self.running:
            return
        if self.transcriber is None:
            self._say("status", "loading model...")
            try:
                from asr import Transcriber

                self.transcriber = Transcriber(self.s)
                if hasattr(self.transcriber, "warmup"):
                    self.transcriber.warmup()
            except Exception as exc:
                self._say("status", f"model failed: {exc}")
                return

        self.segmenter.reset()
        if hasattr(self.vad, "reset"):
            self.vad.reset()

        try:
            self.capture.start()
        except Exception as exc:
            self._say("status", f"audio failed: {exc}")
            return

        self.writer.start_session()
        self.running = True
        self.utterance_q = queue.Queue(maxsize=UTTERANCE_QUEUE_MAX)
        self._asr_worker = threading.Thread(target=self._asr_work, daemon=True)
        self._asr_worker.start()
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()

        self._say("path", str(self.writer.path))
        self._say("badge", self._badge())
        self._say("running", True)
        self._say("status", "listening")

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self.capture.stop()
        if self._worker is not None:
            self._worker.join(timeout=3.0)
            self._worker = None
        # The segmentation thread queues a sentinel on exit; the ASR thread
        # drains remaining utterances (including the flushed tail) before it.
        asr_worker = getattr(self, "_asr_worker", None)
        if asr_worker is not None:
            asr_worker.join(timeout=10.0)
            self._asr_worker = None
        self.writer.stop_session()
        self._say("running", False)
        if self.window is not None:
            self.window.set_level(0.0)

        # Pressing STOP is the natural "I have what I came for" moment, so this
        # is where the transcript is handed over. It covers everything captured
        # since the app launched, not just the stopped session -- the buffer
        # spans toggles by design.
        if getattr(self.s, "copy_on_stop", True):
            self._copy_all_on_stop()
        else:
            self._say("status", "stopped")

    def _copy_all_on_stop(self) -> None:
        try:
            count = self.copy_all()
        except Exception:
            # A broken clipboard must never cost the user a transcript. The
            # text is already durable on disk, so just say the copy failed.
            self._say("status", "stopped - clipboard copy failed")
            return
        if count:
            self._say("status", f"copied {count} line"
                                 f"{'s' if count != 1 else ''} to clipboard")
        else:
            # Either nothing was captured, or the clipboard refused it. The
            # file holds the transcript either way.
            self._say("status", "stopped - nothing to copy")

    def copy_all(self) -> int:
        text = self.buffer.text()
        if not text:
            return 0
        if not sinks.copy_to_clipboard(text):
            return 0
        if self.window is not None:
            self._say("status", f"copied {len(self.buffer)} line"
                                f"{'s' if len(self.buffer) != 1 else ''}"
                                f" to clipboard")
        return len(self.buffer)

    def transcript_path(self) -> str:
        return str(self.writer.path)

    def shutdown(self) -> None:
        try:
            self.stop()
        finally:
            self.writer.stop_session()


def main() -> int:
    app = App()
    from ui import TranscribeWindow

    window = TranscribeWindow(app, app.s)
    app.attach(window)
    window.post("status", f"ready  |  vad: {app.vad_desc}")
    window.install_hotkeys()
    window.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
