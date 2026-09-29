"""WASAPI loopback capture: system audio -> 16 kHz mono frames.

Captures whatever the default output device is playing. The device's native
sample rate is read at open time and never assumed -- on this machine the
Realtek speakers run at 192000 Hz, and a second (Voice.ai) loopback device
also exists, so selection matches the *current default output* by name.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

try:  # pragma: no cover - exercised only on real hardware
    import soxr
except ImportError:  # pragma: no cover
    soxr = None


@dataclass
class Frame:
    """One 512-sample mono frame at the target rate, with its RMS.

    RMS is computed once here so the VU meter and the segmenter's energy
    gate read the same number and cannot diverge.
    """

    samples: np.ndarray
    rms: float


def downmix(block: np.ndarray, channels: int) -> np.ndarray:
    """Average an interleaved N-channel block to mono. Handles any N."""
    if channels <= 1:
        return block.astype(np.float32, copy=False)
    usable = (block.size // channels) * channels
    return block[:usable].reshape(-1, channels).mean(axis=1).astype(np.float32)


def pick_loopback_device(pa) -> dict:
    """Return the loopback device matching the current default output.

    Raises RuntimeError if WASAPI or a matching loopback device is absent.
    """
    import pyaudiowpatch as pyaudio

    try:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError as exc:
        raise RuntimeError("WASAPI host API not available") from exc

    default_out = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
    name = default_out["name"]

    candidates = list(pa.get_loopback_device_info_generator())
    if not candidates:
        raise RuntimeError("No WASAPI loopback devices found")

    for dev in candidates:
        if name in dev["name"]:
            return dev
    # Default output has no loopback twin (rare). Fall back to the first.
    return candidates[0]


class LoopbackCapture:
    """Streams the default output device as 16 kHz mono frames.

    Calls ``on_frame(Frame)`` from the PortAudio callback thread. That
    callback must not block -- it does downmix, resample and framing only.

    A watchdog restarts capture if frames stop arriving, which is what
    happens when the user switches to headphones mid-session (WASAPI
    loopback dies when the default device changes). ``on_device_change``
    is called when that restart happens.
    """

    WATCHDOG_TIMEOUT_S = 2.5

    def __init__(
        self,
        on_frame: Callable[[Frame], None],
        target_rate: int = 16000,
        frame_samples: int = 512,
        pyaudio_factory: Optional[Callable] = None,
        on_device_change: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        self.on_frame = on_frame
        self.target_rate = target_rate
        self.frame_samples = frame_samples
        self.on_device_change = on_device_change
        self.on_error = on_error
        self._pyaudio_factory = pyaudio_factory
        self._pa = None
        self._stream = None
        self._resampler = None
        self._leftover = np.empty(0, dtype=np.float32)
        self._channels = 2
        self._device_rate = target_rate
        self._device_name = ""
        self._running = False
        self._last_frame_at = 0.0
        self._watchdog: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- state

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def device_rate(self) -> int:
        return self._device_rate

    @property
    def running(self) -> bool:
        return self._running

    # ---------------------------------------------------------------- start

    def start(self) -> None:
        if self._running:
            return
        self._open()
        self._running = True
        self._last_frame_at = time.monotonic()
        self._watchdog = threading.Thread(target=self._watch, daemon=True)
        self._watchdog.start()

    def _make_pa(self):
        if self._pyaudio_factory is not None:
            return self._pyaudio_factory()
        import pyaudiowpatch as pyaudio

        return pyaudio.PyAudio()

    def _open(self) -> None:
        import pyaudiowpatch as pyaudio

        self._pa = self._make_pa()
        dev = pick_loopback_device(self._pa)
        self._device_name = dev["name"]
        self._device_rate = int(dev["defaultSampleRate"])
        self._channels = int(dev["maxInputChannels"]) or 2

        self._leftover = np.empty(0, dtype=np.float32)
        if self._device_rate == self.target_rate or soxr is None:
            self._resampler = None
        else:
            self._resampler = soxr.ResampleStream(
                self._device_rate, self.target_rate, 1, dtype="float32", quality="HQ"
            )

        # Chunk sized so each callback carries roughly one output frame.
        chunk = max(256, int(self.frame_samples * self._device_rate / self.target_rate))
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self._channels,
            rate=self._device_rate,
            frames_per_buffer=chunk,
            input=True,
            input_device_index=dev["index"],
            stream_callback=self._callback,
        )
        self._stream.start_stream()

    # ------------------------------------------------------------- callback

    def _callback(self, in_data, frame_count, time_info, status):
        import pyaudiowpatch as pyaudio

        try:
            block = np.frombuffer(in_data, dtype=np.float32)
            mono = downmix(block, self._channels)
            if self._resampler is not None:
                mono = self._resampler.resample_chunk(mono)
            if mono.size:
                self._emit(np.asarray(mono, dtype=np.float32))
            self._last_frame_at = time.monotonic()
        except Exception as exc:  # never raise into PortAudio
            if self.on_error:
                self.on_error(exc)
        return (None, pyaudio.paContinue)

    def _emit(self, mono: np.ndarray) -> None:
        """Accumulate resampled audio and emit fixed-size frames."""
        buf = np.concatenate((self._leftover, mono)) if self._leftover.size else mono
        n = self.frame_samples
        count = buf.size // n
        for i in range(count):
            chunk = buf[i * n : (i + 1) * n]
            rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
            self.on_frame(Frame(samples=chunk.copy(), rms=rms))
        self._leftover = buf[count * n :].copy()

    # ------------------------------------------------------------- watchdog

    def _watch(self) -> None:
        """Restart capture if audio stops arriving (device switched)."""
        while self._running:
            time.sleep(0.5)
            if not self._running:
                return
            silent_for = time.monotonic() - self._last_frame_at
            if silent_for < self.WATCHDOG_TIMEOUT_S:
                continue
            with self._lock:
                if not self._running:
                    return
                previous = self._device_name
                try:
                    self._teardown()
                    self._open()
                    self._last_frame_at = time.monotonic()
                    if self.on_device_change and self._device_name != previous:
                        self.on_device_change(self._device_name)
                except Exception as exc:
                    if self.on_error:
                        self.on_error(exc)
                    time.sleep(2.0)

    # ----------------------------------------------------------------- stop

    def _teardown(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    def stop(self) -> None:
        self._running = False
        with self._lock:
            self._teardown()
