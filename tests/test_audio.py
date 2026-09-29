"""Audio conversion: downmix, framing, RMS, and device selection."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio import Frame, LoopbackCapture, downmix  # noqa: E402


# ---------------------------------------------------------------- downmix

def test_downmix_averages_stereo():
    block = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)  # L,R,L,R
    assert np.allclose(downmix(block, 2), [0.5, 0.5])


def test_downmix_handles_arbitrary_channel_counts():
    block = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    assert np.allclose(downmix(block, 3), [2.0, 5.0])


def test_downmix_passes_mono_through():
    block = np.array([0.1, 0.2], dtype=np.float32)
    assert np.allclose(downmix(block, 1), [0.1, 0.2])


def test_downmix_ignores_a_partial_trailing_frame():
    block = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # 1.5 stereo frames
    assert np.allclose(downmix(block, 2), [1.0])


# ----------------------------------------------------------------- framing

def capture_with(frames_out):
    return LoopbackCapture(on_frame=frames_out.append, target_rate=16000,
                           frame_samples=512)


def test_emits_exactly_sized_frames_and_keeps_the_remainder():
    got = []
    cap = capture_with(got)
    cap._emit(np.ones(512 + 100, dtype=np.float32))
    assert len(got) == 1
    assert got[0].samples.size == 512
    assert cap._leftover.size == 100


def test_remainder_joins_the_next_chunk():
    got = []
    cap = capture_with(got)
    cap._emit(np.ones(300, dtype=np.float32))
    assert got == []
    cap._emit(np.ones(300, dtype=np.float32))
    assert len(got) == 1
    assert cap._leftover.size == 88


def test_emits_multiple_frames_from_a_large_chunk():
    got = []
    cap = capture_with(got)
    cap._emit(np.ones(512 * 3, dtype=np.float32))
    assert len(got) == 3
    assert cap._leftover.size == 0


def test_rms_is_computed_once_and_carried_on_the_frame():
    got = []
    cap = capture_with(got)
    cap._emit(np.full(512, 0.5, dtype=np.float32))
    assert got[0].rms == pytest.approx(0.5, abs=1e-6)


def test_frames_do_not_alias_the_source_buffer():
    got = []
    cap = capture_with(got)
    buf = np.ones(512, dtype=np.float32)
    cap._emit(buf)
    buf[:] = 0.0
    assert got[0].samples[0] == 1.0, "frame must own its samples"


# ------------------------------------------------------- device selection

class FakePA:
    """Minimal stand-in for PyAudio device enumeration."""

    def __init__(self, default_name, loopbacks):
        self._default = default_name
        self._loopbacks = loopbacks

    def get_host_api_info_by_type(self, _t):
        return {"defaultOutputDevice": 0}

    def get_device_info_by_index(self, _i):
        return {"name": self._default}

    def get_loopback_device_info_generator(self):
        return iter(self._loopbacks)


def dev(name, index, rate=192000, ch=2):
    return {"name": name, "index": index, "defaultSampleRate": rate,
            "maxInputChannels": ch}


def test_picks_the_loopback_matching_the_current_default_output():
    import audio

    pa = FakePA("Speakers (Realtek(R) Audio)", [
        dev("Speakers (Realtek(R) Audio) [Loopback]", 13, 192000),
        dev("Speakers (Voice.ai Audio Cable) [Loopback]", 14, 48000),
    ])
    chosen = audio.pick_loopback_device(pa)
    assert chosen["index"] == 13
    assert chosen["defaultSampleRate"] == 192000


def test_follows_the_default_when_it_changes_to_another_device():
    import audio

    pa = FakePA("Speakers (Voice.ai Audio Cable)", [
        dev("Speakers (Realtek(R) Audio) [Loopback]", 13, 192000),
        dev("Speakers (Voice.ai Audio Cable) [Loopback]", 14, 48000),
    ])
    assert audio.pick_loopback_device(pa)["index"] == 14


def test_raises_when_no_loopback_device_exists():
    import audio

    pa = FakePA("Speakers", [])
    with pytest.raises(RuntimeError, match="loopback"):
        audio.pick_loopback_device(pa)
