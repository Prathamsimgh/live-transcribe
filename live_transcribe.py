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

    # ------------------------------------------------------- worker thread

    def _work(self) -> None:
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

    def _emit(self, utterance) -> None:
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
        self.writer.stop_session()
        self._say("running", False)
        self._say("status", "stopped")
        if self.window is not None:
            self.window.set_level(0.0)

    def copy_all(self) -> int:
        text = self.buffer.text()
        if not text:
            return 0
        return len(self.buffer) if sinks.copy_to_clipboard(text) else 0

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
