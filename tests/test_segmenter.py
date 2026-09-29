"""Segmenter boundary behaviour, driven without onnxruntime."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import LOUD, QUIET, frame  # noqa: E402
from segmenter import Segmenter  # noqa: E402


class EnergyVAD:
    """Stand-in VAD: loud frames are speech. Deterministic for tests."""

    def __call__(self, samples):
        return 1.0 if float(np.abs(samples).mean()) > 0.05 else 0.0

    def reset(self):
        pass


def make(settings):
    return Segmenter(settings, EnergyVAD())


def feed(seg, frames):
    out = []
    for f in frames:
        u = seg.push(f)
        if u is not None:
            out.append(u)
    return out


def test_opens_after_speech_start_frames(settings):
    seg = make(settings)
    for _ in range(settings.speech_start_frames - 1):
        assert seg.push(frame(LOUD)) is None
    assert not seg._in_speech
    seg.push(frame(LOUD))
    assert seg._in_speech


def test_preroll_is_prepended_so_first_word_is_not_clipped(settings):
    seg = make(settings)
    # Quiet frames fill the pre-roll ring, then speech opens the utterance.
    feed(seg, [frame(QUIET) for _ in range(settings.preroll_frames)])
    feed(seg, [frame(LOUD) for _ in range(settings.speech_start_frames)])
    # Buffer holds pre-roll plus the speech frames that opened it.
    assert len(seg._buffer) == settings.preroll_frames
    assert len(seg._buffer) > settings.speech_start_frames


def test_closes_after_speech_end_frames_of_silence(settings):
    seg = make(settings)
    feed(seg, [frame(LOUD) for _ in range(40)])
    out = feed(seg, [frame(QUIET) for _ in range(settings.speech_end_frames - 1)])
    assert out == []
    out = feed(seg, [frame(QUIET)])
    assert len(out) == 1
    assert out[0].duration_s > 0


def test_short_blips_are_discarded_as_noise(settings):
    seg = make(settings)
    # 4 loud frames = 128 ms, well under min_utterance_ms of 320.
    feed(seg, [frame(LOUD) for _ in range(4)])
    out = feed(seg, [frame(QUIET) for _ in range(settings.speech_end_frames)])
    assert out == []


def test_soft_cut_lands_on_quietest_trailing_frame(settings):
    settings.max_utterance_s = 1.0
    settings.soft_cut_window_s = 0.5
    seg = make(settings)

    # 32ms frames, so 1.0s of buffer is crossed on the 32nd frame. Feeding
    # only 31 would never trigger the cut.
    n = 32
    dip_index = 25  # inside the trailing 0.5s window (frames 17..31)
    frames = [frame(LOUD) for _ in range(n)]
    frames[dip_index] = frame(0.06)  # quieter, but still speech to the VAD

    out = feed(seg, frames)
    assert len(out) == 1, "max length must force a cut"
    emitted_frames = out[0].samples.size // settings.frame_samples
    assert emitted_frames == dip_index, "cut must land on the quietest frame"


def test_soft_cut_retains_tail_and_stays_in_speech(settings):
    settings.max_utterance_s = 1.0
    settings.soft_cut_window_s = 0.5
    seg = make(settings)

    n = 32
    dip_index = 25
    frames = [frame(LOUD) for _ in range(n)]
    frames[dip_index] = frame(0.06)

    out = feed(seg, frames)
    assert len(out) == 1, "the cut must actually have happened"
    assert seg._in_speech, "must not re-run start detection after a soft cut"
    # Frames from the cut point onward are kept for the next utterance.
    assert len(seg._buffer) == n - dip_index


def test_trailing_silence_is_trimmed_off_the_utterance(settings):
    seg = make(settings)
    feed(seg, [frame(LOUD) for _ in range(40)])
    out = feed(seg, [frame(QUIET) for _ in range(settings.speech_end_frames)])
    assert len(out) == 1
    emitted = out[0].samples.size // settings.frame_samples
    # 40 speech frames plus the short hangover, not all 16 silent frames.
    assert emitted == 40 + settings.trailing_keep_frames


def test_energy_gate_skips_vad_and_scores_zero(settings):
    calls = []

    class Counting:
        def __call__(self, samples):
            calls.append(1)
            return 1.0

        def reset(self):
            pass

    from segmenter import Segmenter as S

    seg = S(settings, Counting())
    for _ in range(50):
        seg.push(frame(QUIET))
    assert calls == [], "silence must never reach the VAD"


def test_mean_vad_prob_is_defined_when_frames_were_gated(settings):
    seg = make(settings)
    feed(seg, [frame(LOUD) for _ in range(40)])
    out = feed(seg, [frame(QUIET) for _ in range(settings.speech_end_frames)])
    assert len(out) == 1
    assert 0.0 <= out[0].mean_vad_prob <= 1.0
    assert not np.isnan(out[0].mean_vad_prob)


def test_flush_closes_in_progress_utterance(settings):
    seg = make(settings)
    feed(seg, [frame(LOUD) for _ in range(40)])
    assert seg._in_speech
    out = seg.flush()
    assert out is not None
    assert not seg._in_speech
    assert seg.flush() is None
