"""Voice activity detection and utterance boundary decisions.

This module decides when a sentence has ended, and is the main determinant
of perceived latency. It emits complete utterances only -- there are no
partial hypotheses, because Whisper rewrites itself when fed truncated
audio.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

# Below this RMS the audio is treated as silence regardless of the adaptive
# noise floor. Guards the case where the floor itself is 0.0 (digital silence),
# which would otherwise let every silent frame through to the VAD.
ABSOLUTE_SILENCE_RMS = 1e-4

# How fast the noise floor tracks. Only non-speech frames feed it, so speech
# cannot drag the floor up to its own level and gate itself out.
NOISE_FLOOR_RATE = 0.05


@dataclass
class Utterance:
    samples: np.ndarray
    mean_vad_prob: float
    duration_s: float


@dataclass
class _F:
    """A frame retained inside the current utterance."""

    samples: np.ndarray
    rms: float
    prob: float


class Segmenter:
    """Turns a stream of frames into complete utterances.

    ``vad`` is any callable taking a frame of ``frame_samples`` float32
    samples and returning a speech probability in [0, 1]. It is injected so
    tests can drive the state machine without onnxruntime.
    """

    def __init__(self, settings, vad: Callable[[np.ndarray], float]) -> None:
        self.s = settings
        self.vad = vad
        self.rate = settings.target_rate
        self.frame_samples = settings.frame_samples
        self.frame_s = settings.frame_samples / settings.target_rate

        # Noise floor, fed only by non-speech frames (see push()).
        self._floor = 0.0

        self._preroll: deque = deque(maxlen=settings.preroll_frames)
        self._buffer: List[_F] = []
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0

    # ------------------------------------------------------------- helpers

    @property
    def noise_floor(self) -> float:
        return self._floor

    def _update_floor(self, rms: float) -> None:
        """Track the floor toward `rms`. Call only for non-speech frames."""
        self._floor += NOISE_FLOOR_RATE * (rms - self._floor)

    def _is_gated(self, rms: float) -> bool:
        """True when the energy pre-gate rejects a frame before the VAD."""
        threshold = max(self._floor * self.s.energy_gate_multiplier,
                        ABSOLUTE_SILENCE_RMS)
        return rms <= threshold

    def _buffer_duration(self) -> float:
        return len(self._buffer) * self.frame_s

    # ---------------------------------------------------------------- push

    def push(self, frame) -> Optional[Utterance]:
        """Feed one frame. Returns an Utterance when a boundary is reached."""
        # Energy pre-gate: skip the VAD entirely on silence. Gated frames
        # count as probability 0.0 so mean_vad_prob stays well defined.
        if self._is_gated(frame.rms):
            prob = 0.0
        else:
            prob = float(self.vad(frame.samples))

        is_speech = prob >= self.s.vad_threshold

        # Only non-speech shapes the noise floor. Feeding it speech would let
        # a long monologue raise the floor above its own level, after which
        # the gate would suppress the speech it is meant to pass through.
        if not is_speech:
            self._update_floor(frame.rms)

        f = _F(samples=frame.samples, rms=frame.rms, prob=prob)

        if not self._in_speech:
            return self._push_idle(f, is_speech)
        return self._push_speech(f, is_speech)

    def _push_idle(self, f: _F, is_speech: bool) -> Optional[Utterance]:
        if is_speech:
            self._speech_run += 1
            self._preroll.append(f)
            if self._speech_run >= self.s.speech_start_frames:
                # Open the utterance with the pre-roll so the first word
                # is not clipped.
                self._buffer = list(self._preroll)
                self._preroll.clear()
                self._in_speech = True
                self._silence_run = 0
                self._speech_run = 0
        else:
            self._speech_run = 0
            self._preroll.append(f)
        return None

    def _push_speech(self, f: _F, is_speech: bool) -> Optional[Utterance]:
        self._buffer.append(f)
        if is_speech:
            self._silence_run = 0
        else:
            self._silence_run += 1
            if self._silence_run >= self.s.speech_end_frames:
                return self._finalize()

        if self._buffer_duration() >= self.s.max_utterance_s:
            return self._soft_cut()
        return None

    # ----------------------------------------------------------- boundaries

    def _trim_trailing_silence(self, frames: List[_F]) -> List[_F]:
        """Drop the detected silence from the tail, keeping a short hangover.

        Without this every utterance carries speech_end_frames of silence:
        it wastes GPU time, invites hallucination on the empty tail, and
        inflates the duration so sub-minimum blips slip through the
        min_utterance_ms filter.
        """
        end = len(frames)
        while end > 0 and frames[end - 1].prob < self.s.vad_threshold:
            end -= 1
        silent = len(frames) - end
        return frames[: end + min(self.s.trailing_keep_frames, silent)]

    def _finalize(self) -> Optional[Utterance]:
        """Close the utterance on confirmed silence."""
        frames = self._trim_trailing_silence(self._buffer)
        self._buffer = []
        self._in_speech = False
        self._silence_run = 0
        self._speech_run = 0
        self._preroll.clear()
        return self._build(frames)

    def _soft_cut(self) -> Optional[Utterance]:
        """Force a cut at max length, choosing the quietest nearby frame.

        Cutting at exactly max_utterance_s would slice a word in half.
        Natural speech nearly always dips within the trailing window, so the
        minimum-RMS frame there is a far better boundary. The tail is
        retained and the segmenter stays in-speech, because re-running start
        detection would discard it and clip the straddling word.
        """
        window_frames = max(1, int(self.s.soft_cut_window_s / self.frame_s))
        start = max(1, len(self._buffer) - window_frames)
        tail = self._buffer[start:]
        offset = int(np.argmin([x.rms for x in tail]))
        cut = start + offset

        head, rest = self._buffer[:cut], self._buffer[cut:]
        self._buffer = rest  # stays in_speech
        self._silence_run = 0
        return self._build(head)

    def _build(self, frames: List[_F]) -> Optional[Utterance]:
        if not frames:
            return None
        duration = len(frames) * self.frame_s
        if duration * 1000.0 < self.s.min_utterance_ms:
            return None  # door clicks, notification blips
        samples = np.concatenate([x.samples for x in frames])
        mean_prob = float(np.mean([x.prob for x in frames])) if frames else 0.0
        return Utterance(samples=samples, mean_vad_prob=mean_prob,
                         duration_s=duration)

    def flush(self) -> Optional[Utterance]:
        """Close any in-progress utterance, e.g. when stopping capture."""
        if not self._in_speech:
            return None
        return self._finalize()

    def reset(self) -> None:
        self._buffer = []
        self._preroll.clear()
        self._floor = 0.0
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
