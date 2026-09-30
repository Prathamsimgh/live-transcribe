"""Optimizations: ASR warmup must not poison context; queue constants sane."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr import Transcriber  # noqa: E402
from config import Settings  # noqa: E402
from segmenter import Utterance  # noqa: E402


class Seg:
    def __init__(self, text):
        self.text = text
        self.avg_logprob = -0.2
        self.no_speech_prob = 0.05


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, samples, **kwargs):
        self.calls.append(kwargs)
        return iter([Seg("real sentence")]), {}


def _factory(model):
    def make(name, device, compute_type):
        return model
    return make


def _utt():
    return Utterance(samples=np.zeros(16000, dtype=np.float32),
                     mean_vad_prob=0.9, duration_s=1.0)


def test_warmup_decodes_without_setting_prompt_context():
    m = FakeModel()
    t = Transcriber(Settings(), model_factory=_factory(m))
    t.warmup()
    assert len(m.calls) == 1
    assert m.calls[0]["beam_size"] == 1
    assert m.calls[0]["condition_on_previous_text"] is False
    # First real utterance still starts with no prompt.
    t.transcribe(_utt())
    assert m.calls[1]["initial_prompt"] is None


def test_utterance_queue_is_bounded():
    import live_transcribe

    assert 1 <= live_transcribe.UTTERANCE_QUEUE_MAX <= 32
